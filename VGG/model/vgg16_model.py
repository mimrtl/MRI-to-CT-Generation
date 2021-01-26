from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model

'''
Description: build VGG16 model
'''

def vgg16_model():
    weights_path = './vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base_model = VGG16(include_top=False, input_shape=(512, 512, 3))
    base_model.load_weights(weights_path)

    # freeze some of the layers
    for layer in base_model.layers[:14]:
    # for layer in base_model.layers[:20]:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid', name='final_pred')(x)

    vgg_model = Model(input=base_model.input, output=predictions)
    return vgg_model
