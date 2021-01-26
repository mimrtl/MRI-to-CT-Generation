import numpy as np
import os
from PIL import Image


# Description: add random intensity map to MR images
def add_intensity_map(path):
    file_list = os.listdir(path)
    for file_name in file_list:
        print(file_name)
        file_path = path + file_name
        ori_img = Image.open(file_path)
        ori_img = np.array(ori_img)
        print(np.max(ori_img), np.min(ori_img))

        # generate the intensity map randomly
        Ax, Ay = 1, 1
        imgx, imgy = 512, 512
        omgx = np.random.rand() * 3e-2
        omgy = np.random.rand() * 3e-2
        phix = np.random.rand() * 1
        phiy = np.random.rand() * 1
        print(omgx, omgy)
        print(phix, phiy)

        x = np.linspace(0, imgx, imgx)
        y = np.linspace(0, imgy, imgy)
        meshx, meshy = np.meshgrid(x, y)
        print(meshx.shape, meshy.shape)
        z = (Ax * np.sin(omgx * meshx + phix) + Ay * np.sin(omgy * meshy + phiy) + 8) / 8
        print('the shape of the intensity map: ', z.shape)
        print(np.max(z), np.min(z))
        print('***************************')

        # add the intensity map to the original image
        aug_img = ori_img * z

        # save the combination of the intensity map and original image
        temp_x = Image.fromarray(aug_img)
        temp_x = temp_x.convert('L')
        out_path = path.split('MR')[0] + 'aug_MR/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        temp_x.save(out_path + 'mask_' + file_name, 'bmp')


if __name__ == '__main__':
    for i in range(1, 11):
        path = '/project/data/bmp_data/per_data/group' + str(i) + '/train/MR/'
        add_intensity_map(path)