"""Train DigitLab's exactly-three-digit CRNN/CTC model on synthetic MNIST strings.

Run this once locally before deploying:
    python train_3digit.py

Output:
    multidigit_ctc.pth (saved beside this script)
"""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_OUT = BASE_DIR / "multidigit_ctc.pth"
TRAIN_SAMPLES = 24000
VALIDATION_SAMPLES = 3000
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TARGET_DIGITS = 3
MAX_WIDTH = 256
BLANK_TOKEN = 10
NUM_CLASSES = 11  # 0-9 plus CTC blank
MEAN = 0.1307
STD = 0.3081

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class MultiDigitCRNN(nn.Module):
    """Keep this architecture identical to the one in app.py."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1)),
        )
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(256, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)
        features = features.squeeze(2).transpose(1, 2)
        sequence, _ = self.rnn(features)
        return self.classifier(sequence).log_softmax(dim=-1).transpose(0, 1)


class SyntheticThreeDigitDataset(Dataset):
    """Combine three labeled MNIST digits with varied width, spacing, and tilt."""

    def __init__(self, mnist_dataset, length: int, seed: int = 0, augment: bool = False):
        self.mnist = mnist_dataset
        self.length = length
        self.seed = seed
        self.augment = augment
        self.affine = transforms.RandomAffine(
            degrees=8,
            translate=(0.07, 0.07),
            scale=(0.86, 1.10),
            shear=6,
            fill=0,
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        # Training examples must vary across accesses/epochs. Previously each index
        # reused the same three MNIST digits throughout training, reducing diversity.
        # Keep validation deterministic so epoch-to-epoch scores remain comparable.
        rng = random if self.augment else random.Random(self.seed + index)
        digit_images: list[np.ndarray] = []
        labels: list[int] = []
        gaps: list[int] = []

        for _ in range(TARGET_DIGITS):
            source_index = rng.randrange(len(self.mnist))
            image, label = self.mnist[source_index]
            if self.augment:
                image = self.affine(image)
                filter_draw = rng.random()
                if filter_draw < 0.10:
                    image = image.filter(ImageFilter.MaxFilter(3))
                elif filter_draw < 0.16:
                    image = image.filter(ImageFilter.MinFilter(3))

            # Real handwriting has wider and narrower digits. Vary tile width
            # without changing the 28px input height used by the CNN.
            width_factor = rng.uniform(0.72, 1.28) if self.augment else rng.uniform(0.82, 1.18)
            tile_width = max(18, min(36, int(round(28 * width_factor))))
            image = image.resize((tile_width, 28), Image.Resampling.BILINEAR)
            digit_images.append(np.asarray(image, dtype=np.uint8))
            labels.append(int(label))

        for _ in range(TARGET_DIGITS - 1):
            gaps.append(rng.randint(1, 9 if self.augment else 7))

        width = sum(tile.shape[1] for tile in digit_images) + sum(gaps)
        if width > MAX_WIDTH:
            raise RuntimeError(f"Generated example too wide ({width}px).")

        # MNIST uses black background and white strokes.
        canvas = np.zeros((28, width), dtype=np.uint8)
        x = 0
        for position, tile in enumerate(digit_images):
            tile_width = tile.shape[1]
            canvas[:, x:x + tile_width] = tile
            x += tile_width
            if position < len(gaps):
                x += gaps[position]

        # Match app.py preprocessing: autocontrast, find the ink bounds, add a
        # small margin, then scale the complete number to 28 px high. The previous
        # trainer fed 28x28 MNIST tiles with their original margins, while inference
        # cropped those margins away. That train/serve mismatch can hurt accuracy.
        processed = ImageOps.autocontrast(Image.fromarray(canvas))
        pixels = np.asarray(processed, dtype=np.uint8)
        ys, xs = np.where(pixels > 35)
        if xs.size == 0 or ys.size == 0:
            raise RuntimeError("Generated an empty synthetic digit sequence.")

        top, bottom = int(ys.min()), int(ys.max()) + 1
        left, right = int(xs.min()), int(xs.max()) + 1
        ink_height = bottom - top
        pad_y = max(2, int(round(ink_height * 0.09)))
        pad_x = max(2, int(round(ink_height * 0.07)))
        top = max(0, top - pad_y)
        bottom = min(28, bottom + pad_y)
        left = max(0, left - pad_x)
        right = min(width, right + pad_x)
        crop = processed.crop((left, top, right, bottom))

        scaled_width = max(1, int(round(crop.width * 28 / crop.height)))
        if scaled_width > MAX_WIDTH:
            raise RuntimeError(f"Generated example too wide after preprocessing ({scaled_width}px).")
        resized = crop.resize((scaled_width, 28), Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - MEAN) / STD
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor, labels, scaled_width


def collate_sequences(batch):
    images, label_lists, widths = zip(*batch)
    batch_size = len(images)
    padded_width = ((max(widths) + 3) // 4) * 4
    normalized_background = (0.0 - MEAN) / STD
    padded = torch.full(
        (batch_size, 1, 28, padded_width),
        fill_value=normalized_background,
        dtype=torch.float32,
    )
    for index, image in enumerate(images):
        padded[index, :, :, :image.shape[-1]] = image

    targets = torch.tensor([label for row in label_lists for label in row], dtype=torch.long)
    target_lengths = torch.tensor([len(row) for row in label_lists], dtype=torch.long)
    input_lengths = torch.tensor([max(1, width // 4) for width in widths], dtype=torch.long)
    return padded, targets, input_lengths, target_lengths, label_lists


def greedy_decode(log_probs: torch.Tensor, input_length: int) -> list[int]:
    """Used only to evaluate exact-sequence accuracy on the validation set."""
    path = log_probs[:input_length].argmax(dim=-1).tolist()
    result: list[int] = []
    previous = BLANK_TOKEN
    for token in path:
        if token != BLANK_TOKEN and token != previous:
            result.append(int(token))
        previous = int(token)
    return result


def run_epoch(model, loader, optimizer, criterion, device, training: bool):
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    exact_correct = 0
    exact_total = 0

    for images, targets, input_lengths, target_lengths, label_lists in loader:
        images = images.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            log_probs = model(images)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        if not training:
            for sample_index in range(images.shape[0]):
                predicted = greedy_decode(log_probs[:, sample_index], int(input_lengths[sample_index]))
                if predicted == label_lists[sample_index]:
                    exact_correct += 1
                exact_total += 1

    mean_loss = total_loss / max(total_batches, 1)
    exact_accuracy = 100.0 * exact_correct / exact_total if exact_total else 0.0
    return mean_loss, exact_accuracy


def main() -> None:
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DigitLab 3-digit model trainer")
    print("Training device:", device)
    print("Only three-digit sequences are generated.")
    print("Downloading/loading MNIST (first run requires internet access)...")

    train_mnist = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    validation_mnist = datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
    train_data = SyntheticThreeDigitDataset(train_mnist, TRAIN_SAMPLES, seed=1000, augment=True)
    validation_data = SyntheticThreeDigitDataset(
        validation_mnist, VALIDATION_SAMPLES, seed=900000, augment=False
    )
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=collate_sequences,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=collate_sequences,
    )

    model = MultiDigitCRNN().to(device)
    criterion = nn.CTCLoss(blank=BLANK_TOKEN, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_accuracy = -1.0
    best_loss = float("inf")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    print(f"Training examples: {TRAIN_SAMPLES}")
    print(f"Validation examples: {VALIDATION_SAMPLES}")
    print(f"Epochs: {EPOCHS}; batch size: {BATCH_SIZE}")
    print("Training begins. Keep this terminal open until training finishes.\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, _ = run_epoch(model, train_loader, optimizer, criterion, device, training=True)
        val_loss, exact_accuracy = run_epoch(
            model, validation_loader, optimizer, criterion, device, training=False
        )
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{EPOCHS} | train CTC loss: {train_loss:.4f} | "
            f"validation CTC loss: {val_loss:.4f} | "
            f"validation exact-sequence accuracy: {exact_accuracy:.2f}% | "
            f"learning rate: {current_lr:.6g}"
        )

        if exact_accuracy > best_accuracy or (
            exact_accuracy == best_accuracy and val_loss < best_loss
        ):
            best_accuracy = exact_accuracy
            best_loss = val_loss
            cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(cpu_state, MODEL_OUT)
            print(f"  Saved best model to {MODEL_OUT.name}")

    print("\nTraining finished.")
    print(f"Best validation exact-sequence accuracy: {best_accuracy:.2f}%")
    print(f"Saved model file: {MODEL_OUT.resolve()}")
    print("Now run: python app.py")
    print("Note: validation accuracy is measured on synthetic MNIST sequences, not every handwriting style.")
    print("Real-image accuracy may still require collecting and training on representative examples.")


if __name__ == "__main__":
    main()
