DigitLab model accuracy improvement

1. Back up your existing train_3digit.py.
2. Replace it with the included train_3digit.py in your DigitLab-3Digit-App folder.
3. Open Command Prompt in that folder and run:
       python train_3digit.py
4. Keep the terminal open until training finishes. This revision uses more epochs and training examples, generates new synthetic digit combinations as training progresses, and aligns training preprocessing with app.py.
5. It overwrites multidigit_ctc.pth with the best model checkpoint.
6. Upload the newly generated multidigit_ctc.pth to the same GitHub repository beside app.py and commit it.
7. Let Render redeploy, then test the live app.

This keeps the same network architecture and checkpoint keys as app.py. It does not guarantee high accuracy on every real-world image. The validation score is measured on synthetic MNIST sequences. For best real-world accuracy, train and evaluate with examples similar to the actual handwriting/photos you will submit.
