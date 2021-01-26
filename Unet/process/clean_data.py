import numpy as np
import nibabel as nib
import os


def clean_ct_data():
    input_folder = '/project/data/nii_data/gp_1/Case19/'
    img_path = input_folder + 'ct_mask.nii'
    img_file = nib.load(img_path)
    img_data = img_file.get_fdata()

    # these are the noises which should be removed
    img_data[img_data <= 50] = 0
    img_data[img_data > 50] = 1

    img_data[437:, :, 60:] = 0
    img_data[:, :130, :] = 0
    img_data[:60, :, :] = 0

    affine = img_file.affine
    header = img_file.header
    nii_file = nib.Nifti1Image(img_data, affine, header)
    nib.save(nii_file, input_folder + 'ct_mask_1.nii')

    ct_path = input_folder + 'CT_clip.nii'
    ct_file = nib.load(ct_path)
    ct_data = ct_file.get_fdata()
    ct_data[ct_data < -1000] = -1000
    ct_data[ct_data > 2000] = 2000
    ct_data += 1000

    new_data = np.zeros((ct_data.shape[0], ct_data.shape[1], ct_data.shape[2]))
    new_data = np.multiply(img_data, ct_data)
    new_data -= 1000
    affine = ct_file.affine
    header = ct_file.header
    nii_file = nib.Nifti1Image(new_data, affine, header)
    nib.save(nii_file, input_folder + 'CT_target.nii')


def clean_mr_data():
    input_folder = '/project/data/nii_data/gp_1/Case19/'
    img_path = input_folder + 'mr_mask.nii'
    img_file = nib.load(img_path)
    img_data = img_file.get_fdata()

    # these are the noises which should be removed
    img_data[img_data <= 30] = 0
    img_data[img_data > 30] = 1

    img_data[448:, :233, 102:111] = 0
    img_data[448:, :, 43:102] = 0
    img_data[443:, :219, 100:105] = 0
    img_data[443:, :, 53:87] = 0
    img_data[443:, :232, 86:101] = 0
    img_data[438:, :206, 45:102] = 0
    img_data[444:, :213, 34:50] = 0
    img_data[431:, :199, 53:70] = 0
    img_data[433:, :183, 94:98] = 0

    # save the clean data
    affine = img_file.affine
    header = img_file.header
    nii_file = nib.Nifti1Image(img_data, affine, header)
    nib.save(nii_file, input_folder + 'mr_mask_1.nii')

    mr_path = input_folder + 'MR_clip.nii'
    mr_file = nib.load(mr_path)
    mr_data = mr_file.get_fdata()

    # new_data = np.zeros((mr_data.shape[0], mr_data.shape[1], mr_data.shape[2]))
    mr_data = np.multiply(img_data, mr_data)

    affine = mr_file.affine
    header = mr_file.header
    nii_file = nib.Nifti1Image(mr_data, affine, header)
    nib.save(nii_file, input_folder + 'MR_target.nii')


if __name__ == '__main__':
    clean_ct_data()
    # clean_mr_data()