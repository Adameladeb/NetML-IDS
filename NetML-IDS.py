import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# Load the preprocessed dataset (features and labels)
data = np.load('UNSW_NB15_training-set.csv', allow_pickle=True)
features = data[:, :-1]  # Extract features (input data)
labels = data[:, -1]  # Extract labels (0 for normal, 1 for intrusion)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Reshape the input data
input_shape = (X_train.shape[1], 1)  # Assuming 1D features
X_train = X_train.reshape(X_train.shape[0], *input_shape)
X_test = X_test.reshape(X_test.shape[0], *input_shape)

# Create the encoder model
encoder_input = tf.keras.Input(shape=input_shape)
encoder = layers.Conv1D(32, 3, activation='relu', padding='same')(encoder_input)
encoder = layers.MaxPooling1D(2, padding='same')(encoder)
encoder = layers.Conv1D(64, 3, activation='relu', padding='same')(encoder)
encoder = layers.MaxPooling1D(2, padding='same')(encoder)
encoder = layers.Conv1D(128, 3, activation='relu', padding='same')(encoder)
encoder = layers.MaxPooling1D(2, padding='same')(encoder)

# Create the decoder model
decoder = layers.Conv1D(128, 3, activation='relu', padding='same')(encoder)
decoder = layers.UpSampling1D(2)(decoder)
decoder = layers.Conv1D(64, 3, activation='relu', padding='same')(decoder)
decoder = layers.UpSampling1D(2)(decoder)
decoder = layers.Conv1D(32, 3, activation='relu', padding='same')(decoder)
decoder = layers.UpSampling1D(2)(decoder)
decoder_output = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(decoder)

# Combine the encoder and decoder into an autoencoder model
autoencoder = tf.keras.Model(inputs=encoder_input, outputs=decoder_output)

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder model
autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, validation_data=(X_test, X_test))

# Obtain the learned features from the encoder part
encoder_model = tf.keras.Model(inputs=encoder_input, outputs=encoder)
encoded_features_train = encoder_model.predict(X_train)
encoded_features_test = encoder_model.predict(X_test)

# Create the classifier model
classifier_input = tf.keras.Input(shape=encoder.shape[1:])
classifier = layers.Flatten()(classifier_input)
classifier = layers.Dense(64, activation='relu')(classifier)
classifier_output = layers.Dense(1, activation='sigmoid')(classifier)

# Combine the encoder and classifier into a classification model
classification_model = tf.keras.Model(inputs=classifier_input, outputs=classifier_output)

# Compile the classification model
classification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the classification model
classification_model.fit(encoded_features_train, y_train, epochs=10, batch_size=32, validation_data=(encoded_features_test, y_test))

# Evaluate the classification model
loss, accuracy = classification_model.evaluate(encoded_features_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
