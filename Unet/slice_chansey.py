#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import glob
import datetime
import argparse
import numpy as np
from global_dict.w_global import gbl_set_value, gbl_get_value, gbl_save_value
from process.MRCT_load import write_XY
from process.combine import load_x_y
from model.slice_train import train_a_unet
from predict.MRCT_predict import predict_MRCT

np.random.seed(591)


def usage():
    print("Error in input argv")


def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for sCT generation from low filed MRI images. ''',
        epilog="""All's well that ends well.""")

    parser.add_argument('--slice_x', metavar='', type=int, default=1,
                        help='channel of input(1)<int>[1/3]')
    parser.add_argument('--input_size', metavar='', type=int, default=512,
                        help='The size of the input image, eg 224 and 512')
    parser.add_argument('--id', metavar='', type=str, default="chansey",
                        help='ID of the current model.(eeVee)<str>')
    parser.add_argument('--epoch', metavar='', type=int, default=1000,
                        help='Number of epoches of training(300)<int>')
    parser.add_argument('--n_filter', metavar='', type=int, default=64,
                        help='The initial filter number of Unet(64)<int>')
    parser.add_argument('--depth', metavar='', type=int, default=4,
                        help='The depth of Unet(4)<int>')
    parser.add_argument('--batch_size', metavar='', type=int, default=4,
                        help='The batch_size of training(10)<int>')
    parser.add_argument('--tv_weight', metavar='', type=float, default=1e-4,
                        help='record the tv_weight')
    parser.add_argument('--content_weight', metavar='', type=float, default=2500,
                        help='record the content_weight')
    parser.add_argument('--style_weight', metavar='', type=float, default=11000,
                        help='record the style_weight')
    parser.add_argument('--binary_weight', metavar='', type=float, default=100,
                        help='record the style_weight')
    parser.add_argument('--group_id', metavar='', type=int, default=1,
                        help='The id of groups(1)<int>[1/2/3/4/5/6/7/8/9/10]')
    parser.add_argument('--round_id', metavar='', type=str, default='1st_round',
                        help='The id of rounds(1)<str>[1st_round/2nd_round]')
    parser.add_argument('--case_id', metavar='', type=int, default=10,
                        help='The number pf cases used for training(10)<int>')
    parser.add_argument('--pretrained_flag', metavar='', type=int, default=0,
                        help='The flag of pretrained model or new model(0)<int>')
    parser.add_argument('--pretrained_path', metavar='', type=str, default="---the pretrained path---",
                        help='The path of pretrained model<str>')
    parser.add_argument('--discriminator_path', metavar='', type=str,
                        default='/code/data/per/gp_1/10-cases/1st_round/dis/model/loss_model_chansey-2020-07-15-23-10.hdf5',
                        help='the discriminator path')

    args = parser.parse_args()

    data_path = '/code/data/per/gp_' + str(args.group_id) + '/' + str(args.case_id) + '-cases/data_for_gen/'

    train_num = len(os.listdir(data_path + '/train/train_y/CT/'))
    val_num = len(os.listdir(data_path + '/val/val_y/CT/'))

    dir_syn = '/code/data/per/gp_' + str(args.group_id) + '/' + str(args.case_id) + '-cases/' + args.round_id + '/gen/model/'
    dir_model = '/code/data/per/gp_' + str(args.group_id) + '/' + str(args.case_id) + '-cases/' + args.round_id + '/gen/model/'

    if not os.path.exists(dir_syn):
        os.makedirs(dir_syn)

    if not os.path.exists(dir_model):
        os.makedirs(dir_model)

    time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    model_id = args.id + time_stamp
    gbl_set_value("depth", args.depth)
    gbl_set_value("dir_syn", dir_syn)
    gbl_set_value("dir_model", dir_model)
    gbl_set_value("model_id", model_id)
    gbl_set_value("n_epoch", args.epoch + 1)
    gbl_set_value("n_filter", args.n_filter)
    gbl_set_value("depth", args.depth)
    gbl_set_value("batch_size", args.batch_size)
    gbl_set_value("slice_x", args.slice_x)
    gbl_set_value("n_slice_train", train_num)
    gbl_set_value("n_slice_val", val_num)
    gbl_set_value('tv_weight', args.tv_weight)
    gbl_set_value('content_weight', args.content_weight)
    gbl_set_value('style_weight', args.style_weight)
    gbl_set_value('binary_weight', args.binary_weight)
    gbl_set_value('discriminator_path', args.discriminator_path)
    gbl_set_value('input_size', args.input_size)
    gbl_set_value("pretrained_flag", args.pretrained_flag)
    gbl_set_value("pretrained_path", args.pretrained_path)

    model = train_a_unet(data_path)
    print("Training Completed!")


if __name__ == "__main__":
    main()


