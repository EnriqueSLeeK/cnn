
import utils.file_operation as fop
import tensorflow as tf
from tensorflow import keras

import numpy as np


# Uma classe para guardar o loss/cost do modelo
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# Funcao para tratar os dados
def get_data_preprocessed():
    mnist = keras.datasets.mnist

    (train_images, train_labels), (test_images, test_labels) = \
        mnist.load_data()

# Vetorizacao das imagens
    row_size = 28
    col_size = 28
    if keras.backend.image_data_format() == 'channels_first':
        train_images = train_images.reshape(train_images.shape[0],
                                            1,
                                            row_size,
                                            col_size)
        test_images = test_images.reshape(test_images.shape[0],
                                          1,
                                          row_size,
                                          col_size)
        input_shape = (1,
                       row_size,
                       col_size)
    else:
        train_images = train_images.reshape(train_images.shape[0],
                                            row_size,
                                            col_size,
                                            1)
        test_images = test_images.reshape(test_images.shape[0],
                                          row_size,
                                          col_size,
                                          1)
        input_shape = (row_size,
                       col_size,
                       1)
# Normalizacao das imagens
    train_images, test_images = train_images / 255.0, test_images / 255.0

# Transformacao da label para um array 1d de categorias para ficar do jeito
# do softmax
    num_category = 10
    train_labels = keras.utils.to_categorical(train_labels, num_category)
    test_labels = keras.utils.to_categorical(test_labels, num_category)

    return ((train_images, train_labels),
            (test_images, test_labels),
            input_shape)


# Relu foi a funcao que mais ajudou a diminuir o erro do modelo
def create_model(filters_count=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding_mode="valid",
                 act="relu",
                 input_shapes=(28, 28, 1),
                 pool_sizes=(2, 2),
                 hidden_dense=16,
                 output_dense=10,
                 output_act="softmax"):
    model = keras.models.Sequential()


# A primeira camada recebe o parametro input_shape porque vai
# ser a camada de entrada
# O tamanho do kernel ficou com o tamanho 3x3 foi o melhor
# e o stride(deslizamento do kernel) vai dar um passo
    model.add(keras.layers.Conv2D(filters=filters_count,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding_mode,
                                  activation=act,
                                  input_shape=input_shapes))
# O max pooling serve para "resumir" as informacoes obtidas
    model.add(keras.layers.MaxPool2D(pool_size=pool_sizes))

# A mesmo procedimento
    model.add(keras.layers.Conv2D(filters=filters_count,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding_mode,
                                  activation=act))
    model.add(keras.layers.MaxPool2D(pool_size=pool_sizes))

# No final vamos usar "achatar" a matriz
# e usar um layer denso semelhante ao usado na mlp
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hidden_dense,
                                 activation=act))

# Camada de saida usando a funcao passada no output_act
# nesse caso e o softmax
    model.add(keras.layers.Dense(output_dense,
                                 activation=output_act))

    return model


if __name__ == "__main__":

    (train_images, train_labels),\
            (test_images, test_labels),\
            input_shape = get_data_preprocessed()

    model = create_model(input_shapes=input_shape)

# Compilar o modelo
# Usando o optimizador Adam
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])

    loss_history = LossHistory()

# Treinamento
    model.fit(train_images,
              train_labels,
              epochs=10,
              batch_size=64,
              verbose=2,
              callbacks=[loss_history])

# Teste
    model.evaluate(test_images,
                   test_labels,
                   batch_size=64,
                   verbose=2)

    predictions = model.predict(test_images,
                                batch_size=64,
                                verbose=2)

# A matriz de confusao
    confusion_matrix = tf.math.confusion_matrix(
            labels=np.argmax(test_labels, axis=1),
            predictions=np.argmax(predictions, axis=1))
