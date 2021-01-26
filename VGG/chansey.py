#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import datetime
import argparse
import numpy as np
from global_dict.w_global import gbl_set_value, gbl_get_value, gbl_save_value
from model.w_train import train_a_unet
from process.data_load import data_load
from process.channel import channel

np.random.seed(591)


def usage():
    print("Error in input argv")


def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for the discriminator between CT and sCT. ''',
        epilog="""All's well that ends well.""")
    parser.add_argument('--slice_x', metavar='', type=int, default=3,
                        help='dimension of input(1)<int>[1/3]')
    parser.add_argument('--n_pixel', metavar='', type=int, default=512,
                        help='data size of input(1)<int>[512]')
    parser.add_argument('--id', metavar='', type=str, default="chansey",
                        help='ID of the current model.(eeVee)<str>')
    parser.add_argument('--epoch', metavar='', type=int, default=12000,
                        help='Number of epoches of training(300)<int>')
    parser.add_argument('--batch_size', metavar='', type=int, default=16,
                        help='The batch_size of training(16)<int>')
    parser.add_argument('--group_id', metavar='', type=int, default=1,
                        help='The id of groups(1)<int>[1/2/3/4/5/6/7/8/9/10]')
    parser.add_argument('--case_id', metavar='', type=int, default=10,
                        help='The number pf cases used for training(10)<int>[10/20/30]')
    parser.add_argument('--round_id', metavar='', type=str, default='1st_round',
                        help='The id of groups(1st_round)<str>[1st_round/2nd_round]')
    parser.add_argument('--pretrained_flag', metavar='', type=int, default=0,
                        help='The flag of pretrained model or new model(0)<int>[0/1]')
    parser.add_argument('--pretrained_path', metavar='', type=str, default="",
                        help='The path of pretrained model')
    args = parser.parse_args()

    dir_syn = '/code/data/per/gp_' + str(args.group_id) + '/' + str(args.case_id) + '-cases/' + args.round_id + '/dis/model/'
    dir_model = '/code/data/per/gp_' + str(args.group_id) + '/' + str(args.case_id) + '-cases/' + args.round_id + '/dis/model/'
    print(dir_syn)

    if not os.path.exists(dir_syn):
        os.makedirs(dir_syn)
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)

    train_path = '/code/data/per/gp_' + str(args.group_id) + '/' + str(args.case_id) + '-cases/data_for_dis/train/'
    val_path = '/code/data/per/gp_' + str(args.group_id) + '/' + str(
        args.case_id) + '-cases/data_for_dis/val/'

    train_num = os.listdir(train_path + 'CT/')
    val_num = os.listdir(val_path + 'CT/')
    n_slice = (len(train_num) + len(val_num)) * 2

    time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    model_id = args.id + time_stamp
    gbl_set_value("dir_syn", dir_syn)
    gbl_set_value("dir_model", dir_model)
    gbl_set_value("model_id", model_id)
    gbl_set_value("n_slice", n_slice)
    gbl_set_value("n_epoch", args.epoch + 1)
    gbl_set_value("batch_size", args.batch_size)
    gbl_set_value("slice_x", args.slice_x)
    gbl_set_value("n_pixel", args.n_pixel)
    gbl_set_value("pretrained_flag", args.pretrained_flag)
    gbl_set_value("pretrained_path", args.pretrained_path)

    # print(train_x.shape, train_y.shape)
    print("Loading Completed!")

    model = train_a_unet(train_path, val_path)
    print("Training Completed!")


if __name__ == "__main__":
    main()
