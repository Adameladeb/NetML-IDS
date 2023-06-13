# Import necessary libraries
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks, applications
from sklearn.ensemble import IsolationForest
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import shap

# Load the preprocessed dataset (features and labels)
data = np.genfromtxt('UNSW_NB15_training-set.csv', delimiter=',', skip_header=1)
features = data[:, :-1]  # Extract features (input data)
labels = data[:, -1]  # Extract labels (0 for normal, 1 for intrusion)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize the input data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Reshape the input data
input_shape = (X_train.shape[1], 1)  # Assuming 1D features
X_train = X_train.reshape(X_train.shape[0], *input_shape)
X_test = X_test.reshape(X_test.shape[0], *input_shape)

# Create the encoder model
encoder_input = tf.keras.Input(shape=input_shape)
encoder = layers.Conv1D(32, 3, activation='relu', padding='same')(encoder_input)
encoder = layers.BatchNormalization()(encoder)
encoder = layers.MaxPooling1D(2, padding='same')(encoder)
encoder = layers.Conv1D(64, 3, activation='relu', padding='same')(encoder)
encoder = layers.BatchNormalization()(encoder)
encoder = layers.MaxPooling1D(2, padding='same')(encoder)
encoder = layers.Conv1D(128, 3, activation='relu', padding='same')(encoder)
encoder = layers.BatchNormalization()(encoder)
encoder = layers.MaxPooling1D(2, padding='same')(encoder)

# Create the decoder model
decoder = layers.Conv1D(128, 3, activation='relu', padding='same')(encoder)
decoder = layers.BatchNormalization()(decoder)
decoder = layers.UpSampling1D(2)(decoder)
decoder = layers.Conv1D(64, 3, activation='relu', padding='same')(decoder)
decoder = layers.BatchNormalization()(decoder)
decoder = layers.UpSampling1D(2)(decoder)
decoder = layers.Conv1D(32, 3, activation='relu', padding='same')(decoder)
decoder = layers.BatchNormalization()(decoder)
decoder = layers.UpSampling1D(2)(decoder)
decoder_output = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(decoder)

# Combine the encoder and decoder into an autoencoder model
autoencoder = tf.keras.Model(inputs=encoder_input, outputs=decoder_output)

# Plot the autoencoder model
plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True, show_layer_names=True)

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Create a custom callback to monitor the training process
class MonitorCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('Epoch:', epoch+1, 'Loss:', logs['loss'], 'Val Loss:', logs['val_loss'])

# Train the autoencoder model without data augmentation
autoencoder.fit(X_train, X_train, batch_size=32, epochs=10, validation_data=(X_test, X_test), callbacks=[MonitorCallback()])

# Obtain the learned features from the encoder part
encoder_model = tf.keras.Model(inputs=encoder_input, outputs=encoder)
encoded_features_train = encoder_model.predict(X_train)
encoded_features_test = encoder_model.predict(X_test)

# Apply anomaly detection using Isolation Forest
anomaly_detector = IsolationForest(contamination=0.05)
anomaly_detector.fit(encoded_features_train)

# Predict anomalies in the test set
anomaly_predictions = anomaly_detector.predict(encoded_features_test)
anomaly_predictions = np.where(anomaly_predictions == -1, 1, 0)

# Fine-tune a pre-trained ResNet model for anomaly detection
resnet = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in resnet.layers:
    layer.trainable = False
x = layers.Flatten()(resnet.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation='sigmoid')(x)
anomaly_detector = models.Model(inputs=resnet.input, outputs=x)

# Plot the anomaly detector model
plot_model(anomalydetector, to_file='anomaly_detector.png', show_shapes=True, show_layer_names=True)

# Compile the anomaly detector model
anomaly_detector.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the anomaly detector model with data augmentation
# Note: Data augmentation is not necessary for this specific task of anomaly detection
# You can remove it if you want.
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

anomaly_detector.fit(datagen.flow(X_train, y_train, batch_size=32),
                     steps_per_epoch=len(X_train) // 32, epochs=10,
                     validation_data=(X_test, y_test), callbacks=[MonitorCallback()])

# Evaluate the anomaly detector model
loss, accuracy = anomaly_detector.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Use SHAP to explain the predictions of the anomaly detector model
# Note: You need to provide more details on how you want to use SHAP to explain the predictions
explainer = shap.Explainer(anomaly_detector)
shap_values = explainer(X_test)

# Create an ensemble model of the autoencoder and the anomaly detector
anomaly_detector2 = models.Model(inputs=resnet.input, outputs=x)  # Define the second anomaly detector
anomaly_detector3 = models.Model(inputs=resnet.input, outputs=x)  # Define the third anomaly detector
ensemble_input = tf.keras.Input(shape=input_shape)
encoded_features = encoder_model(ensemble_input)
anomaly_output1 = anomaly_detector(encoded_features)
anomaly_output2 = anomaly_detector2(encoded_features)
anomaly_output3 = anomaly_detector3(encoded_features)
ensemble_output = layers.Average()([anomaly_output1, anomaly_output2, anomaly_output3])
ensemble_model = models.Model(inputs=ensemble_input, outputs=ensemble_output)

# Plot the ensemble model
plot_model(ensemble_model, to_file='ensemble_model.png', show_shapes=True, show_layer_names=True)

# Compile the ensemble model
ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the ensemble model
loss, accuracy = ensemble_model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
