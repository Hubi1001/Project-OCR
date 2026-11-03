# Polish Character Recognition: a CNN on Synthetic Data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hubi1001/Project-OCR/blob/main/Projekt_OCR.ipynb)

A small convolutional network that classifies 28x28 grayscale images of 74 characters: the digits 0-9 and the Polish alphabet in lower and upper case. The images are generated inside the notebook, so no dataset has to be downloaded.

**What this is not:** the data is synthetic. Each image is a character rendered from a TrueType font (the generator tries DejaVu Sans first) with random size, rotation, shift, noise and blur. It is not real handwriting, and the test set comes from the same generator, so the accuracy below says little about performance on real handwritten text.

## Setup and result

| Item | Value |
|---|---|
| Classes | 74 (10 digits, 32 lowercase and 32 uppercase letters) |
| Data | 7,400 generated images (100 per class); stratified split into 5,920 training and 1,480 test images; 15% of the training set is used for validation |
| Model | 5 convolutional layers with batch normalization and dropout, two dense layers (256 and 128 units), softmax output; 479,274 parameters |
| Training | Adam (learning rate 0.001), batch size 64, up to 30 epochs, early stopping on validation loss (patience 7), learning-rate reduction on plateau |
| Random seed | 42 |
| Test accuracy | 90.07% on the 1,480 synthetic test images |
| Environment | TensorFlow 2.21.0, Keras 3.15.1, CPU |

## Notes and limitations

- A fixed seed is set at the top of the notebook. Results can still differ between hardware and library versions.
- The augmentation in section 7 is only previewed. The model is trained without it.
- The generator needs one of the fonts listed in the notebook. If none is found, it silently falls back to the default PIL font, which changes the data completely. A diagnostic cell near the top prints which font is available.
- The error analysis is partial: the confusion matrix shows only the first 20 classes.

## How to run

In Colab, click the badge above. Locally:

```bash
git clone https://github.com/Hubi1001/Project-OCR.git
cd Project-OCR
pip install tensorflow numpy matplotlib pillow scikit-learn seaborn jupyter
jupyter notebook Projekt_OCR.ipynb
```

The notebook saves the trained model to the working directory (`best_polish_handwriting_model.keras`, `polish_handwriting_final.keras`, `model_architecture.json`, `class_mapping.npy`). These files are not stored in the repository.

## Contact

Hubert Poździoch - [LinkedIn](https://www.linkedin.com/in/hubert-pozdzioch-4a5933297/) - [GitHub](https://github.com/Hubi1001)
