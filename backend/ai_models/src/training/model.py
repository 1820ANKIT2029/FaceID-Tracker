import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import RandomNormal

import heapq

from ..config import config

IM_SIZE = config["IM_SIZE"]
REGULARIZATION_RATE = config["REGULARIZATION_RATE"]
DROPOUT_RATE_CONV = config["DROPOUT_RATE_CONV"]
DROPOUT_RATE_DENSE = config["DROPOUT_RATE_DENSE"]

class EmbeddingModel(Model):
    def __init__(self, **kwargs):
        super(EmbeddingModel, self).__init__(name='embedding', **kwargs)

        self.conv1 = Conv2D(
            filters=64,
            kernel_size=(10, 10),
            activation=None,
            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),
            bias_initializer=RandomNormal(mean=0.5, stddev=1e-2),
            kernel_regularizer=L2(REGULARIZATION_RATE)
        )
        self.bn1 = BatchNormalization()
        self.act1 = ReLU()
        self.pool1 = MaxPooling2D((2, 2), strides=2, padding='same')
        self.drop1 = Dropout(DROPOUT_RATE_CONV)

        self.conv2 = Conv2D(
            filters=128,
            kernel_size=(7, 7),
            activation=None,
            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),
            bias_initializer=RandomNormal(mean=0.5, stddev=1e-2),
            kernel_regularizer=L2(REGULARIZATION_RATE)
        )
        self.bn2 = BatchNormalization()
        self.act2 = ReLU()
        self.pool2 = MaxPooling2D((2, 2), strides=2, padding='same')
        self.drop2 = Dropout(DROPOUT_RATE_CONV)

        self.conv3 = Conv2D(
            filters=128,
            kernel_size=(4, 4),
            activation=None,
            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),
            bias_initializer=RandomNormal(mean=0.5, stddev=1e-2),
            kernel_regularizer=L2(REGULARIZATION_RATE)
        )
        self.bn3 = BatchNormalization()
        self.act3 = ReLU()
        self.pool3 = MaxPooling2D((2, 2), strides=2, padding='same')
        self.drop3 = Dropout(DROPOUT_RATE_CONV)

        self.conv4 = Conv2D(
            filters=256,
            kernel_size=(4, 4),
            activation=None,
            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),
            bias_initializer=RandomNormal(mean=0.5, stddev=1e-2),
            kernel_regularizer=L2(REGULARIZATION_RATE)
        )
        self.bn4 = BatchNormalization()
        self.act4 = ReLU()
        self.flatten = Flatten()
        self.drop4 = Dropout(DROPOUT_RATE_CONV)

        self.dense = Dense(
            4096,
            activation=None,
            kernel_initializer=RandomNormal(mean=0.0, stddev=2e-1),
            bias_initializer=RandomNormal(mean=0.5, stddev=1e-2),
            kernel_regularizer=L2(REGULARIZATION_RATE)
        )
        self.bn_dense = BatchNormalization()
        self.act_dense = ReLU()
        self.drop5 = Dropout(DROPOUT_RATE_DENSE)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.flatten(x)
        x = self.drop4(x)

        x = self.dense(x)
        x = self.bn_dense(x)
        x = self.act_dense(x)
        x = self.drop5(x)

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

        self.classifier = Dense(
            1,
            activation='sigmoid',
            kernel_initializer=RandomNormal(mean=0.0, stddev=2e-1),
            bias_initializer=RandomNormal(mean=0.5, stddev=1e-2),
            kernel_regularizer=L2(REGULARIZATION_RATE)
        )
        self.drop_classifier = Dropout(DROPOUT_RATE_DENSE)

    def call(self, inputs):
        input_image, validation_image = inputs[0], inputs[1]

        # Get embeddings
        input_embedding = self.embedding(input_image)
        validation_embedding = self.embedding(validation_image)

        # Calculate distance
        distance = self.l1_distance(input_embedding, validation_embedding)

        # Classification
        output = self.classifier(distance)
        output = self.drop_classifier(output)

        return output

    def get_embedding_vector(self, img_name_list):
        output = []

        embedding = None

        for name, img in img_name_list:
            img = tf.expand_dims(img, axis=0)
            embedding_vec = self.embedding.predict(img)

            output.append((name, embedding_vec))

        return output

    def custom_prediction(self, input_img, val_img_embedding, verbose=False):
        """
        val_img_embedding = [
            ("Ankit kumar", [[33, 4, 4, 534 ...]]),
            ("Anup kumar", [[33, 4, 4, 534 ...]]),
            ("Ankit kumar", [[33, 4, 4, 534 ...]]),
        ]
        """

        input_img = tf.expand_dims(input_img, axis=0)  # reshape (100, 100, 3) -> (1, 100, 100, 3)
        person = self.embedding.predict(input_img)

        if verbose:
            print("Person Embedding vector: ", person)

        pq = []

        for name, val_emb in val_img_embedding:
            dist = self.l1_distance(person, val_emb)
            output = self.classifier(dist)

            if verbose:
                print("L1 dist vector: ", dist)
                print("Classifier Output: ", output)

            output =  output.numpy().item()

            if len(pq) < 3:
                heapq.heappush(pq, (output, name))
            else:
                p, n = heapq.heappop(pq)
                if output > p:
                    heapq.heappush(pq, (output, name))
                else:
                    heapq.heappush(pq, (p, n))

        value = []
        while pq:
            value.append(heapq.heappop(pq))

        value.reverse()

        return value

custom_objects = {'SiameseModel': SiameseModel, 'EmbeddingModel': EmbeddingModel, 'L1Dist': L1Dist}