# -*- coding: utf-8 -*-
"""


@author: VMoiseienko
"""

from abc import ABC, abstractmethod
import numpy as np
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier

# Define the DigitClassificationInterface
class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, image):
        pass

# CNN Model
class CNNModel(DigitClassificationInterface):
    def __init__(self):
        self.model = self.build_cnn_model()

    def build_cnn_model(self):
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28, 1)),
            keras.layers.Conv2D(32, kernel_size=(4, 4), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_input(self, image):
        return image  # For CNN, the input is already in the correct format

    def predict(self, image):
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = self.preprocess_input(image)
        return np.argmax(self.model.predict(image), axis=1)[0]

# Random Forest Model
class RandomForestModel(DigitClassificationInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def preprocess_input(self, image):
        return image.reshape(-1)

    def predict(self, image):
        image = self.preprocess_input(image)
        return self.model.predict([image])[0]

# Random Model
class RandomModel(DigitClassificationInterface):
    def predict(self, image):
        return np.random.randint(0, 10)

# Digit Classifier
class DigitClassifier:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.model = self.get_model_instance()

    def get_model_instance(self):
        if self.algorithm == 'cnn':
            return CNNModel()
        elif self.algorithm == 'rf':
            return RandomForestModel()
        elif self.algorithm == 'rand':
            return RandomModel()
        else:
            raise ValueError("Invalid algorithm. Choose 'cnn', 'rf', or 'rand'.")

    def predict(self, image):
        return self.model.predict(image)

# Example usage:
# digit_classifier = DigitClassifier('cnn')
# prediction = digit_classifier.predict(your_28x28x1_image)
# print(prediction)