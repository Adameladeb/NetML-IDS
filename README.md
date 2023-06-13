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

Anomaly Detection using Autoencoder and ResNet50 Models

This code implements an anomaly detection system using two models: an autoencoder and a fine-tuned ResNet50 model.
Dataset

The dataset used in this code is the UNSW-NB15 dataset, which contains network traffic data for intrusion detection. The dataset is preprocessed and split into training and testing sets using a 80:20 ratio.
Autoencoder Model

The autoencoder model is used to learn the features of the input data. The learned features are then used for anomaly detection using Isolation Forest. The architecture of the autoencoder model is as follows:

    Input layer: 1D convolutional layer with 32 filters, kernel size of 3, and ReLU activation function.
    Batch normalization layer.
    Max pooling layer with pool size of 2.
    1D convolutional layer with 64 filters, kernel size of 3, and ReLU activation function.
    Batch normalization layer.
    Max pooling layer with pool size of 2.
    1D convolutional layer with 128 filters, kernel size of 3, and ReLU activation function.
    Batch normalization layer.
    Max pooling layer with pool size of 2.
    1D convolutional layer with 128 filters, kernel size of 3, and ReLU activation function.
    Batch normalization layer.
    Up-sampling layer with size of 2.
    1D convolutional layer with 64 filters, kernel size of 3, and ReLU activation function.
    Batch normalization layer.
    Up-sampling layer with size of 2.
    1D convolutional layer with 32 filters, kernel size of 3, and ReLU activation function.
    Batch normalization layer.
    Up-sampling layer with size of 2.
    Output layer: 1D convolutional layer with 1 filter, kernel size of 3, and sigmoid activation function.

The autoencoder model is trained for 10 epochs using binary cross-entropy loss and the Adam optimizer. The learned features are obtained from the encoder part of the autoencoder model.
Anomaly Detection using Isolation Forest

The learned features are used as input to the Isolation Forest algorithm for anomaly detection. The algorithm is used to predict anomalies in the test set with a 5% contamination rate.
Fine-tuned ResNet50 Model

The ResNet50 model is fine-tuned for anomaly detection using the learned features from the autoencoder model. The architecture of the model is as follows:

    Input layer: ResNet50 with pre-trained weights on ImageNet dataset.
    Flatten layer.
    Dense layer with 256 units and ReLU activation function.
    Dropout layer with rate of 0.5.
    Output layer: Dense layer with 1 unit and sigmoid activation function.

The ResNet50 model is trained for 10 epochs using binary cross-entropy loss and the Adam optimizer. The model is fine-tuned on the learned features from the autoencoder model for anomaly detection.
Ensemble Model

An ensemble model is created by combining the autoencoder and ResNet50 models. The ensemble model takes the input data and generates features using the autoencoder model. The features are then used as input to the ResNet50 model and two additional ResNet50 models to generate predictions. The predictions from the three models are then averaged to generate the final output of the ensemble model.

The ensemble model is evaluated on the test set using binary cross-entropy loss and accuracy as the evaluation metrics.

## Model Overview
The code implements an autoencoder-based anomaly detection and classification model. The autoencoder is trained to reconstruct the input data and learn useful representations in its hidden layers. The learned features from the encoder part are then used as inputs to a separate classifier model for binary classification (normal vs. intrusion).

The autoencoder consists of convolutional layers to extract features and learn their representations. The decoder part reconstructs the input based on the learned features. The classification model takes the learned features as input and uses dense layers to perform binary classification.

Both the autoencoder and classification models are trained separately. The autoencoder is trained to minimize the binary cross-entropy loss between the input and reconstructed output. The classification model is trained to minimize the binary cross-entropy loss and maximize accuracy.

The evaluation of the classification model provides insights into its performance on the testing data, including the calculated loss and accuracy metrics.

Feel free to modify and experiment with the code according to your specific requirements. ✨

If you want to use the code give me credits pls W乇乇り#9249
