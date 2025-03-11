import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.saving import register_keras_serializable

from ..config import config

IM_SIZE = config["IM_SIZE"]

class EmbeddingModel(Model):
    def __init__(self, **kwargs):
        super(EmbeddingModel, self).__init__(name='embedding', **kwargs)

        # First block
        self.conv1 = Conv2D(64, (10, 10), activation='relu')
        self.pool1 = MaxPooling2D((2, 2), padding='same')

        # Second block
        self.conv2 = Conv2D(128, (7, 7), activation='relu')
        self.pool2 = MaxPooling2D((2, 2), padding='same')

        # Third block
        self.conv3 = Conv2D(128, (4, 4), activation='relu')
        self.pool3 = MaxPooling2D((2, 2), padding='same')

        # Final embedding block
        self.conv4 = Conv2D(256, (4, 4), activation='relu')
        self.flatten = Flatten()
        self.dense = Dense(4096, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x


# Siamese L1 Distance class
class L1Dist(Layer):
    def __init__(self):
        super(L1Dist, self).__init__(name='distance')

    def call(self, anchor_embedding, validation_embedding):
        return tf.math.abs(anchor_embedding - validation_embedding)

@register_keras_serializable()
class SiameseModel(Model):
    def __init__(self, **kwargs):
        super(SiameseModel, self).__init__(**kwargs)

        self.embedding = EmbeddingModel()

        self.l1_distance = L1Dist()

        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        input_image, validation_image = inputs[0], inputs[1]

        # input_a = Input(shape=(IM_SIZE, IM_SIZE, 3))
        # input_b = Input(shape=(IM_SIZE, IM_SIZE, 3))
        # Get embeddings
        input_embedding = self.embedding(input_image)
        validation_embedding = self.embedding(validation_image)

        # Calculate distance
        distance = self.l1_distance(input_embedding, validation_embedding)

        # Classification
        output = self.classifier(distance)

        return output

    # def prediction(self, input_img, val_img_arr):
    #     person = self.embedding(input_img)

    #     name = None
    #     max_ = -1
    #     """
    #     val_img_arr = (
    #         ("Ankit kumar", [33, 4, 4, 534 ...]),
    #         ("Anup kumar", [33, 4, 4, 534 ...]),
    #         ("Ankit kumar", [33, 4, 4, 534 ...]),
    #     )
    #     """
    #     for key, img_vec in val_img_arr:
    #         t = some(img_vec)

    #         dist = self.l1_distance(person, t)

    #         output = self.classifier(dist)

    #         if output > max_:
    #             max_ = output
    #             name = key

    #     return (name, max_)

custom_objects = {'SiameseModel': SiameseModel, 'EmbeddingModel': EmbeddingModel, 'L1Dist': L1Dist}