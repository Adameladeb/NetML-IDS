# NetML-IDS

📚 This README provides an overview and usage instructions for the NetML-IDS. 🖥️

## Description
The provided code demonstrates the implementation of an autoencoder-based anomaly detection and classification model using TensorFlow. The code performs the following steps:

1. Load the preprocessed dataset (`UNSW_NB15_training-set.csv`) consisting of features and labels.
2. Split the dataset into training and testing sets.
3. Reshape the input data to prepare it for the autoencoder model.
4. Create the encoder model using 1D convolutional layers and max pooling.
5. Create the decoder model using convolutional layers and upsampling.
6. Combine the encoder and decoder to create an autoencoder model.
7. Compile and train the autoencoder model using the training data.
8. Obtain the learned features from the encoder part of the autoencoder.
9. Create the classifier model using dense layers.
10. Combine the encoder and classifier to create a classification model.
11. Compile and train the classification model using the learned features and corresponding labels.
12. Evaluate the classification model using the testing data and calculate the loss and accuracy.
13. Print the test loss and test accuracy. ✔️

## Requirements
To run the code, the following dependencies are required:
- NumPy (`pip install numpy`)
- TensorFlow (`pip install tensorflow`)
- scikit-learn (`pip install scikit-learn`)

## Usage
1. Ensure that the dataset file `UNSW_NB15_training-set.csv` is located in the same directory as the code file.
2. Install the required dependencies if not already installed.
3. Execute the code in a Python environment.

⚠️ Note: It is assumed that the dataset file is correctly preprocessed and formatted with features and labels. Adjustments may be necessary to handle different datasets or preprocessing steps.

## Model Overview
The code implements an autoencoder-based anomaly detection and classification model. The autoencoder is trained to reconstruct the input data and learn useful representations in its hidden layers. The learned features from the encoder part are then used as inputs to a separate classifier model for binary classification (normal vs. intrusion).

The autoencoder consists of convolutional layers to extract features and learn their representations. The decoder part reconstructs the input based on the learned features. The classification model takes the learned features as input and uses dense layers to perform binary classification.

Both the autoencoder and classification models are trained separately. The autoencoder is trained to minimize the binary cross-entropy loss between the input and reconstructed output. The classification model is trained to minimize the binary cross-entropy loss and maximize accuracy.

The evaluation of the classification model provides insights into its performance on the testing data, including the calculated loss and accuracy metrics.

Feel free to modify and experiment with the code according to your specific requirements. ✨

If you want to use the code give me credits pls W乇乇り#9249
