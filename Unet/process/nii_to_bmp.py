import numpy as np
import nibabel as nib
import os
from PIL import Image
from skimage import transform
import copy
import shutil
import matplotlib.pyplot as plt
import cv2
from process.normalization import MaxMinNorm


# Description: convert Nifti file to bmp file (slices)
def nii_to_bmp(gp_id, type):
    path_in = '/project/data/nii_data/group' + str(gp_id) + '/train/'
    g = os.walk(path_in)
    num = 255
    for path, dir_list, file_list in g:
        if len(file_list) > 0:
            for file in file_list:
                if type in file:
                    data = nib.load(path + '/' + file)
                    data = data.get_fdata()

                    dir_sct = '/project/data/bmp_data/group' + str(gp_id) + '/train/' + type + '/'
                    if not os.path.exists(dir_sct):
                        os.makedirs(dir_sct)

                    data = MaxMinNorm(data, type)
                    out_path_x = dir_sct + path.split('/')[-1] + '_' + type + '_z_'

                    for j in range(data.shape[-1]):
                        temp_x = np.array((512, 512), dtype=float)
                        temp_x = copy.deepcopy(data[:, :, j])
                        temp_x *= num
                        temp_x = Image.fromarray(temp_x)
                        temp_x = temp_x.convert('L')
                        temp_x.save(out_path_x + str(j) + '.bmp', 'bmp')
                else:
                    pass


if __name__ == '__main__':
    # the type to be CT or MR
    type = 'CT'
    for i in range(1, 11):
        print(i)
        nii_to_bmp(i, type)





