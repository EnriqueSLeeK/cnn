
import utils.create_dict as c_dict
from tensorflow import keras
import cnn

hyper_param = c_dict.extract_tup_dict("tf_model_param", "hyper_param.txt")

def preprocess_run():

    (train_images, train_labels),\
        (test_images, test_labels),\
        input_shape = cnn.get_data_preprocessed()

    model = cnn.create_model(
            filters_count=hyper_param['filters'],
            kernel_size=hyper_param['kernel_size'],
            strides=hyper_param['stride'],
            padding_mode=hyper_param['padding_mode'],
            act=hyper_param['activation'],
            input_shapes=hyper_param['input_size'],
            pool_sizes=hyper_param['pooling_size'],
            hidden_dense=hyper_param['hidden_layer'],
            output_dense=hyper_param['output_layer'],
            output_act=hyper_param['output_act']
            )

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])

    model.load_weights('tf_model_param/final_weight/final_weights_param')\
        .expect_partial()

    model.evaluate(test_images,
                   test_labels,
                   batch_size=64,
                   verbose=2)

    # model.predict(test_images,
    #             batch_size=64,
    #             verbose=2)


def train_with_predetermined_param():

    (train_images, train_labels),\
        (test_images, test_labels),\
        input_shape = cnn.get_data_preprocessed()

    model = cnn.create_model(
            filters_count=hyper_param['filters'],
            kernel_size=hyper_param['kernel_size'],
            strides=hyper_param['stride'],
            padding_mode=hyper_param['padding_mode'],
            act=hyper_param['activation'],
            input_shapes=hyper_param['input_size'],
            pool_sizes=hyper_param['pooling_size'],
            hidden_dense=hyper_param['hidden_layer'],
            output_dense=hyper_param['output_layer'],
            output_act=hyper_param['output_act']
            )

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])

    model.load_weights('tf_model_param/init_weight/init_weights_param')\
        .expect_partial()

    model.fit(train_images,
              train_labels,
              epochs=10,
              batch_size=64,
              verbose=2)

    model.evaluate(test_images,
                   test_labels,
                   batch_size=64,
                   verbose=2)


# Para testar o modelo ja treinado com os parametros no e-disciplinas
# Descomentar preprocess_run()
# Para treinar com os parametros iniciais
# Descomentar train_with_predetermined_param()
if __name__ == "__main__":
    preprocess_run()
    # train_with_predetermined_param()
    pass
