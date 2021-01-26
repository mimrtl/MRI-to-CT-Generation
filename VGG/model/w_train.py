#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from global_dict.w_global import gbl_get_value
from model.vgg16_model import vgg16_model
from keras import optimizers
from keras.models import load_model
from keras_radam import RAdam
# from keras.utils import multi_gpu_model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SEED = 314

'''
Description: train VGG16 model
'''


def train_a_unet(train_path, val_path):
    slice_x = gbl_get_value("slice_x")
    n_pixel = gbl_get_value("n_pixel")
    n_slice = gbl_get_value("n_slice")
    model_id = gbl_get_value("model_id")
    dir_model = gbl_get_value('dir_model')
    epochs = gbl_get_value("n_epoch")
    batch_size = gbl_get_value("batch_size")
    flag_save = True

    # ----------------------------------------------Configurations----------------------------------------------#

    # save the log
    log_path = dir_model.split('model')[0] + 'log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    tensorboard = TensorBoard(log_dir=log_path, batch_size=batch_size,
                              write_graph=True, write_grads=True,
                              write_images=True)

    # set traininig configurations
    conf = {"image_shape": (n_pixel, n_pixel, slice_x), "out_channel": 1, "shuffle": True, "augmentation": True,
            "learning_rate": 1e-4, "validation_split": 0.3, "batch_size": batch_size, "epochs": epochs,
            "model_id": model_id}
    np.save(log_path + model_id + '_info.npy', conf)

    # set augmentation configurations
    conf_a = {"rotation_range": 15, "shear_range": 10,
              "width_shift_range": 0.33, "height_shift_range": 0.33, "zoom_range": 0.33,
              "horizontal_flip": True, "vertical_flip": True, "fill_mode": 'nearest',
              "seed": 314, "batch_size": conf["batch_size"]}
    np.save(log_path + model_id + '__aug.npy', conf_a)

    # save the models which has the minimum validation loss or highest accuracy
    if flag_save:
        check_path1 = dir_model+'loss_model_'+model_id+'.hdf5'
        checkpoint1 = ModelCheckpoint(check_path1, monitor='val_loss',
                                      verbose=1, save_best_only=True, save_weights_only=False, mode='min')
        check_path2 = dir_model+'acc_model_'+model_id+'.hdf5'
        checkpoint2 = ModelCheckpoint(check_path2, monitor='val_acc',
                                      verbose=1, save_best_only=True, save_weights_only=False, mode='max')
        callbacks_list = [checkpoint1, checkpoint2, tensorboard]
    else:
        callbacks_list = [tensorboard]

    # ----------------------------------------------Create Model----------------------------------------------#
    # build up the model
    if gbl_get_value("pretrained_flag") == 0:
        model = vgg16_model()
        # model = multi_gpu_model(model, 2)
    else:
        model_path = gbl_get_value("pretrained_path")
        model = load_model(model_path)

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

    # validation data_generator
    data_generator3 = ImageDataGenerator(rescale=1./255,
                                         width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])
                                         # preprocessing_function=aug_noise)
    aug_dir = ''

    # ----------------------------------------------Train Model----------------------------------------------#

    # compile
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_crossentropy', 'acc'])

    train_path = train_path
    val_path = val_path

    # train
    model.fit_generator(generator=data_generator1.flow_from_directory(directory=train_path, target_size=(512, 512),
                                                       color_mode='rgb', classes=None, class_mode="binary",
                                                       batch_size=conf_a["batch_size"], seed=conf_a["seed"],
                                                       shuffle=True, save_to_dir=aug_dir, save_prefix='train'),
                        steps_per_epoch=int(int(n_slice * (1-conf["validation_split"])) / conf_a["batch_size"]),
                        epochs=conf["epochs"],
                        callbacks=callbacks_list,
                        validation_data=data_generator3.flow_from_directory(directory=val_path, target_size=(512, 512),
                                                             color_mode='rgb', classes=None, class_mode="binary",
                                                             batch_size=conf_a["batch_size"], seed=conf_a["seed"],
                                                             shuffle=True, save_to_dir=aug_dir, save_prefix='val'),
                        validation_steps=int(int(n_slice * conf["validation_split"]) / conf_a["batch_size"]))

    return model

