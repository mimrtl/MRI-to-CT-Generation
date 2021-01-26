import os
import random
import shutil


# Description: make paired MR and CT slices to be in one folder
def combine_mr_ct(in_path, tar_path):
    path = in_path
    path_dir = os.listdir(path)
    for name in path_dir:
        tar_dir = tar_path
        ori_name = name.split('mask_')[-1]
        temp = ori_name.split('.')
        temp_1 = temp[0].split('_')
        # MR: 2 and 3; CT: 1 and 2
        new_dir = temp_1[0] + '_' + temp_1[2] + '_' + temp_1[3]
        tar_dir = tar_dir + new_dir
        if not os.path.exists(tar_dir):
            os.makedirs(tar_dir)
        shutil.copyfile(path + name, tar_dir + '/' + name)


# Description: split the paired slices into two parts, train and validation.
def random_data(in_path):
    file_dir = in_path
    tar_dir_val = in_path.split('paired')[0] + 'train_val/' + 'val_pair/'
    tar_dir_train = in_path.split('paired')[0] + 'train_val/' + 'train_pair/'

    if not os.path.exists(tar_dir_train):
        os.makedirs(tar_dir_train)
    if not os.path.exists(tar_dir_val):
        os.makedirs(tar_dir_val)

    path_dir = os.listdir(file_dir)
    file_num = len(path_dir)
    rate = 0.3
    pick_num = int(file_num*rate)
    random.shuffle(path_dir)
    val_sample = random.sample(path_dir, pick_num)
    train_sample = [i for i in path_dir if i not in val_sample]

    for name in val_sample:
        shutil.move(file_dir+name, tar_dir_val+name)
    for name in train_sample:
        shutil.move(file_dir+name, tar_dir_train+name)
    return tar_dir_train, tar_dir_val


# Description: separate the paired MR and CT images into two folders, MR and CT.
def sep_mr_ct(in_path, mr_path, ct_path):
    path = in_path
    case = os.listdir(path)
    sct_path = mr_path
    ct_path = ct_path
    # sct_path = sct_path
    for i in case:
        file_list = os.listdir(path+i)
        for file in file_list:
            print(file)
            print(path+i+'/'+file)
            if 'MR' in file:
                shutil.copyfile(path + i + '/' + file, sct_path + file)
            elif 'CT' in file:
                shutil.copyfile(path + i + '/' + file, ct_path + file)
            else:
                raise RuntimeError('something else appears: {}'.format(file))


if __name__ == '__main__':
    for i in range(1, 11):
        path = '/project/data/bmp_data/groups/group' + str(i) + '/train/train_bmp'
        dir_mr = path + '/aug_MR/'
        dir_ct = path + '/CT/'
        tar_path = path + '/paired/'

        combine_mr_ct(dir_mr, tar_path)
        combine_mr_ct(dir_ct, tar_path)

        tar_dir_train, tar_dir_val = random_data(tar_path)

        train_mr_path = tar_dir_train.split('_pair')[0] + '/train_x/aug_MR/'
        train_ct_path = tar_dir_train.split('_pair')[0] + '/train_y/CT/'

        val_mr_path = tar_dir_val.split('_pair')[0] + '/val_x/aug_MR/'
        val_ct_path = tar_dir_val.split('_pair')[0] + '/val_y/CT/'

        if not os.path.exists(train_mr_path):
            os.makedirs(train_mr_path)
        if not os.path.exists(train_ct_path):
            os.makedirs(train_ct_path)
        if not os.path.exists(val_mr_path):
            os.makedirs(val_mr_path)
        if not os.path.exists(val_ct_path):
            os.makedirs(val_ct_path)

        sep_mr_ct(tar_dir_train, train_mr_path, train_ct_path)
        sep_mr_ct(tar_dir_val, val_mr_path, val_ct_path)





