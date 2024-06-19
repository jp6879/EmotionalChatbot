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

    img_height = 48
    img_width = 48
    num_classes_reduced = 4

    # Agregamos a un modelo VGG hecho los bloques CBAM
    input_img = layers.Input(shape = (img_height, img_width, 1), name = 'imagenes')

    # Bloque 1
    x = layers.Conv2D(64, (3, 3), padding='same')(input_img)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = CBAMBlock(64, reduction_ratio=32)(x)

    x = layers.MaxPooling2D((2, 2), strides=2)(x)

    # Bloque 2
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = CBAMBlock(128, reduction_ratio=32)(x)

    x = layers.MaxPooling2D((2, 2), strides=2)(x)

    # Bloque 3
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = CBAMBlock(256, reduction_ratio=64)(x)

    x = layers.MaxPooling2D((2, 2), strides=2)(x)


    # Bloque 4
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = layers.MaxPooling2D((2, 2), strides=2)(x)

    x = CBAMBlock(512, reduction_ratio=64)(x)

    # Flatten
    x = layers.Flatten()(x)

    # Capas fully connected
    x = layers.Dense(4096)(x)
    x = tf.nn.relu(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(4096)(x)
    x = tf.nn.relu(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(num_classes_reduced, activation = 'softmax')(x)

    # Creamos el modelo
    classifier = keras.models.Model(inputs=input_img, outputs=output)

    # Cargamos los pesos del modelo
    classifier.load_weights(r'C:\Users\Propietario\Desktop\ib\ProyectoV2\EmotionalChatbot\models\ModelCBAM_red.h5')

    return classifier