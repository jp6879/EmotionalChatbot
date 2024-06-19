import tensorflow as tf
import keras
from keras import layers
from keras.models import load_model


def CreateModel():
    # Implementación de CBAM en Keras
    @keras.saving.register_keras_serializable()
    class CBAMBlock(layers.Layer):
        def __init__(self, filter_num, reduction_ratio=32, kernel_size=7, **kwargs):
            super(CBAMBlock, self).__init__(**kwargs)
            self.filter_num = filter_num
            self.reduction_ratio = reduction_ratio
            self.kernel_size = kernel_size

            self.global_avg_pool = layers.GlobalAveragePooling2D()
            self.global_max_pool = layers.GlobalMaxPool2D()
            self.dense1 = layers.Dense(self.filter_num // self.reduction_ratio, activation='relu')
            self.dense2 = layers.Dense(self.filter_num)
            self.sigmoid = layers.Activation('sigmoid')
            self.reshape = layers.Reshape((1, 1, self.filter_num))
            self.multiply = layers.Multiply()

            self.conv2d = layers.Conv2D(1, kernel_size=self.kernel_size, padding='same')
            self.spatial_sigmoid = layers.Activation('sigmoid')

        def call(self, input_tensor):
            axis = -1

            avg_pool = self.global_avg_pool(input_tensor)
            max_pool = self.global_max_pool(input_tensor)
            avg_out = self.dense2(self.dense1(avg_pool))
            max_out = self.dense2(self.dense1(max_pool))
            channel = self.sigmoid(avg_out + max_out)
            channel = self.reshape(channel)
            channel_out = self.multiply([input_tensor, channel])

            avg_pool2 = tf.reduce_mean(input_tensor, axis=axis, keepdims=True)
            max_pool2 = tf.reduce_max(input_tensor, axis=axis, keepdims=True)
            spatial = layers.concatenate([avg_pool2, max_pool2], axis=axis)
            spatial_out = self.spatial_sigmoid(self.conv2d(spatial))

            cbam_out = self.multiply([channel_out, spatial_out])
            return cbam_out

    classifier = keras.models.load_model('../EmotionalChatbot/models/ModelCBAM.keras', custom_objects={'CBAMBlock': CBAMBlock})

    return classifier