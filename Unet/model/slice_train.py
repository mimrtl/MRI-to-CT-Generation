#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from global_dict.w_global import gbl_get_value, gbl_set_value
from model.unet import unet
from keras_radam import RAdam
from keras.models import load_model
import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.utils import multi_gpu_model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SEED = 314


# define mean square error
def mean_squared_error_12(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[1, 2])


def mean_squared_error_123(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[1, 2, 3])


# to make the value larger, mean square error multiplies 1e6
def mean_squared_error_1e6(y_true, y_pred):
    loss = K.mean(K.square(y_pred - y_true), axis=[0, 1, 2, 3])*1e6
    # reg_term = K.square(K.sum(y_pred) - K.sum(y_true))
    return loss


# define mean absolute error
def mean_absolute_error_1e6(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


# compute the gram matrix which is required to compute the style loss
def gram_matrix(x):
    if K.ndim(x) == 4:
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        features = K.reshape(x, K.stack([B, C, H*W]))
        gram = K.batch_dot(features, features, axes=2)
        denominator = C * H * W
        gram = gram / K.cast(denominator, x.dtype)
        return gram
    else:
        raise ValueError('the dimension is not correct!')


# Description: compute style loss
def style_loss(y_pred, y_true):
    style_weight = gbl_get_value('style_weight')
    batch_size = gbl_get_value("batch_size")
    discriminator_path = gbl_get_value('discriminator_path')
    input_size = gbl_get_value('input_size')

    print(style_weight)
    y_true = K.reshape(y_true, (-1, input_size, input_size, 3))
    y_pred = K.reshape(y_pred, (-1, input_size, input_size, 3))

    print('+++++++++++++++++++++++++++++++++++++')
    print('style_loss')
    print(y_true.shape)
    print(y_pred.shape)
    print('+++++++++++++++++++++++++++++++++++++')

    # load the model
    model_path = discriminator_path
    model = load_model(model_path, compile=False)
    for layer in model.layers:
        layer.trainable = False

    selected_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'final_pred']
    selected_output = [model.get_layer(name).output for name in selected_layers]
    layer_model = Model(inputs=model.input, outputs=selected_output)

    feature_pred = layer_model(y_pred)
    feature_true = layer_model(y_true)

    # feature_gram
    gram_pred = []
    gram_true = []
    for i in range(len(feature_pred) - 1):
        gram_pred.append(gram_matrix(feature_pred[i]))
        gram_true.append(gram_matrix(feature_true[i]))

    style_loss = 0
    for i in range(len(selected_layers) - 1):
        temp = mean_squared_error_12(gram_true[i], gram_pred[i][:batch_size])
        style_loss += temp
    style_loss = style_weight * style_loss

    return style_loss


# Description: compute content loss
def content_loss(y_true, y_pred):
    content_weight = gbl_get_value('content_weight')
    discriminator_path = gbl_get_value('discriminator_path')
    input_size = gbl_get_value('input_size')
    y_true = K.reshape(y_true, (-1, input_size, input_size, 3))

    print('+++++++++++++++++++++++++++++++++++++')
    print('content_loss')
    print(y_true.shape)
    print(y_pred.shape)
    print('+++++++++++++++++++++++++++++++++++++')

    # load the model
    model_path = discriminator_path
    model = load_model(model_path, compile=False)
    for layer in model.layers:
        layer.trainable = False

    selected_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'final_pred']
    selected_output = [model.get_layer(name).output for name in selected_layers]
    layer_model = Model(inputs=model.input, outputs=selected_output)

    feature_pred = layer_model(y_pred)
    feature_true = layer_model(y_true)

    content_loss = mean_squared_error_123(feature_true[2], feature_pred[2])
    content_loss = content_weight * content_loss

    return content_loss


# regularization part
def tv_loss(y_true, y_pred):

    tv_weight = gbl_get_value('tv_weight')

    diff_i = K.sum(K.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]))
    diff_j = K.sum(K.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))

    tv_loss = tv_weight * (diff_i + diff_j)

    return tv_loss


# Description: compute perceptual loss
def perceptual_loss(y_true, y_pred):

    batch_size = gbl_get_value("batch_size")
    style_weight = gbl_get_value('style_weight')
    content_weight = gbl_get_value('content_weight')
    tv_weight = gbl_get_value('tv_weight')
    binary_weight = gbl_get_value('binary_weight')
    discriminator_path = gbl_get_value('discriminator_path')
    input_size = gbl_get_value('input_size')

    y_true = K.reshape(y_true, (-1, input_size, input_size, 3))
    mse_loss = mean_squared_error_1e6(y_true, y_pred)

    print('+++++++++++++++++++++++++++++++++++++')
    print('perceptual_loss')
    print(y_true.shape)
    print(y_pred.shape)
    print('+++++++++++++++++++++++++++++++++++++')

    # load the model
    model_path = discriminator_path
    model = load_model(model_path, compile=False)
    for layer in model.layers:
        layer.trainable = False

    selected_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'final_pred']
    selected_output = [model.get_layer(name).output for name in selected_layers]
    layer_model = Model(inputs=model.input, outputs=selected_output)

    feature_pred = layer_model(y_pred)
    feature_true = layer_model(y_true)

    # feature_gram
    gram_pred = []
    gram_true = []
    for i in range(len(feature_pred)-1):
        gram_pred.append(gram_matrix(feature_pred[i]))
        gram_true.append(gram_matrix(feature_true[i]))

    one = tf.ones_like(feature_true[-1])
    feature_true[-1] = tf.where(feature_true[-1] >= 0, x=one, y=one)

    style_loss = 0
    for i in range(len(selected_layers)-1):
        temp = mean_squared_error_12(gram_true[i], gram_pred[i][:batch_size])
        style_loss += temp
    style_loss = style_weight * style_loss

    content_loss = mean_squared_error_123(feature_true[2], feature_pred[2])
    content_loss = content_weight * content_loss

    diff_i = K.sum(K.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]))
    diff_j = K.sum(K.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))

    tv_loss = tv_weight * (diff_i + diff_j)

    final_percep_loss = style_loss + content_loss + tv_loss

    return final_percep_loss

    # the following loss will be returned in the 2nd round of model training with 33 or 34 cases
    # return final_percep_loss + mse_loss


def psnr(y_true, y_pred):
    max_pixel = 1.0
    psnr = (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303
    return K.minimum(psnr, 100)


def train_a_unet(data_path):

    slice_x = gbl_get_value("slice_x")
    n_slice_train = gbl_get_value("n_slice_train")
    n_slice_val = gbl_get_value("n_slice_val")
    n_pixel = gbl_get_value('input_size')
    model_id = gbl_get_value("model_id")
    dir_model = gbl_get_value('dir_model')

    epochs = gbl_get_value("n_epoch")
    n_fliter = gbl_get_value("n_filter")
    depth = gbl_get_value("depth")
    batch_size = gbl_get_value("batch_size")
    optimizer = 'RAdam'
    flag_save = True


    # ----------------------------------------------Configurations----------------------------------------------#

    # save logs
    log_path = dir_model.split('model')[0] + 'log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    tensorboard = TensorBoard(log_dir=log_path, batch_size=batch_size,
                              write_graph=True, write_grads=True,
                              write_images=True)

    # set traininig configurations
    conf = {"image_shape": (n_pixel, n_pixel, slice_x), "out_channel": 1, "filter": n_fliter, "depth": depth,
            "inc_rate": 2, "activation": 'relu', "dropout": True, "batchnorm": True, "maxpool": True,
            "upconv": True, "residual": True, "shuffle": True, "augmentation": True,
            "learning_rate": 1e-4, "decay": 0.0, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999,
            "validation_split": 0.3, "batch_size": batch_size, "epochs": epochs,
            "loss": "mse1e6", "metric": "mse", "optimizer": optimizer, "model_id": model_id}
    np.save(log_path + model_id + '_info.npy', conf)

    # set augmentation configurations
    conf_a = {"rotation_range": 15, "shear_range": 10,
              "width_shift_range": 0.33, "height_shift_range": 0.33, "zoom_range": 0.33,
              "horizontal_flip": True, "vertical_flip": True, "fill_mode": 'nearest',
              "seed": 314, "batch_size": conf["batch_size"]}
    np.save(log_path + model_id + '__aug.npy', conf_a)

    if flag_save:
        check_path_1 = dir_model+'psnr_model_'+model_id+'.hdf5'  # _{epoch:03d}_{val_loss:.4f}
        checkpoint1 = ModelCheckpoint(check_path_1, monitor='val_psnr',
                                      verbose=1, save_best_only=True, mode='max')
        check_path_2 = dir_model+'loss_model_'+model_id+'.hdf5'  # _{epoch:03d}_{val_loss:.4f}
        checkpoint2 = ModelCheckpoint(check_path_2, monitor='val_loss',
                                      verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint1, checkpoint2, tensorboard]
    else:
        callbacks_list = [tensorboard]

    # ----------------------------------------------Create Model----------------------------------------------#

    # build up the model
    print(conf)

    if gbl_get_value("pretrained_flag") == 0:
        print('-----------------start with new model---------------')
        model = unet(img_shape=conf["image_shape"], out_ch=conf["out_channel"],
                     start_ch=conf["filter"], depth=conf["depth"],
                     inc_rate=conf["inc_rate"], activation=conf["activation"],
                     dropout=conf["dropout"], batchnorm=conf["batchnorm"],
                     maxpool=conf["maxpool"], upconv=conf["upconv"],
                     residual=conf["residual"])
        # model = multi_gpu_model(model, 3)

    else:
        # load model
        print('-----------------fine tune previous models----------------')
        model_path = gbl_get_value("pretrained_path")
        model = load_model(model_path, compile=False)

    # for the perceptual loss model, the loss function is perceptual loss
    loss = perceptual_loss

    # for the mse loss model, the loss function is mse loss
    # loss = mean_squared_error_1e6

    opt = RAdam(learning_rate=conf['learning_rate'], total_steps=10000, warmup_proportion=0.1, min_lr=1e-6)

    # ----------------------------------------------Data Generator----------------------------------------------#

    # train data_generator
    data_generator1 = ImageDataGenerator(rescale=1./255,
                                         rotation_range=conf_a["rotation_range"],
                                         shear_range=conf_a["shear_range"],
                                         width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])
                                         # preprocessing_function=aug_noise)
    data_generator2 = ImageDataGenerator(rescale=1./255,
                                         rotation_range=conf_a["rotation_range"],
                                         shear_range=conf_a["shear_range"],
                                         width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])
                                         # preprocessing_function=aug_noise)


    # validation data_generator
    data_generator3 = ImageDataGenerator(rescale=1./255,
                                         width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])
                                         # preprocessing_function=aug_noise)
    data_generator4 = ImageDataGenerator(rescale=1./255,
                                         width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])
                                         # preprocessing_function=aug_noise)

    aug_dir = ''

    train_x_path = data_path + 'train/train_x/'
    train_y_path = data_path + 'train/train_y/'
    val_x_path = data_path + 'val/val_x/'
    val_y_path = data_path + 'val/val_y'

    # zip files
    data_generator_t = zip(data_generator1.flow_from_directory(train_x_path, target_size=(512, 512),
                                                color_mode='grayscale', classes=None, class_mode=None,
                                                batch_size=conf_a["batch_size"], shuffle=True, seed=conf_a["seed"],
                                                save_to_dir=aug_dir, save_prefix='train_x'),
                           data_generator2.flow_from_directory(train_y_path, target_size=(512, 512),
                                                color_mode='rgb', classes=None, class_mode=None,
                                                batch_size=conf_a["batch_size"], shuffle=True, seed=conf_a["seed"],
                                                save_to_dir=aug_dir, save_prefix='train_y'))

    data_generator_v = zip(data_generator3.flow_from_directory(val_x_path, target_size=(512, 512),
                                                color_mode='grayscale', classes=None, class_mode=None,
                                                batch_size=conf_a["batch_size"], shuffle=True, seed=conf_a["seed"],
                                                save_to_dir=aug_dir, save_prefix='val_x'),
                           data_generator4.flow_from_directory(val_y_path, target_size=(512, 512),
                                                color_mode='rgb', classes=None, class_mode=None,
                                                batch_size=conf_a["batch_size"], shuffle=True, seed=conf_a["seed"],
                                                save_to_dir=aug_dir, save_prefix='val_y'))


    # ----------------------------------------------Train Model----------------------------------------------#

    # compile
    model.compile(loss=loss, optimizer=opt, metrics=[content_loss, style_loss, psnr, mean_squared_error_1e6, mean_absolute_error_1e6])

    # train
    model.fit_generator(generator=data_generator_t,
                        steps_per_epoch=int(n_slice_train / conf_a["batch_size"]),
                        epochs=conf["epochs"],
                        callbacks=callbacks_list,
                        validation_data=data_generator_v,
                        validation_steps=int(n_slice_val / conf_a["batch_size"]))  #

    return model
