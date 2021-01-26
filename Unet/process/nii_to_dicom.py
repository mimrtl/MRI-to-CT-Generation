import pydicom
import os
import nibabel as nib
import numpy as np


# Description: convert Nifti file to Dicom files
def nii_dcm_auto():
    g = os.walk("/project/data/nii_data/gp_1/test/")
    for path, dir_list, file_list in g:
        if len(dir_list) == 0 and len(file_list) > 0:
            for cs_file in file_list:
                if 're' not in cs_file or 'Store' in cs_file:
                    continue
                print(cs_file)
                modality = 'CT'
                if 'MR' in cs_file:
                    modality = 'Align_MR'
                print("path: " + path)
                sct_path = ''
                sct_path = path + '/' + cs_file
                print("sct-path:" + sct_path)
                sct_file = nib.load(sct_path)
                sct_data = sct_file.get_fdata()

                # get the original CT in Dicom format
                dir_path = path.split('nii')[0] + 'dcm' + path.split('nii')[1] + '/' + modality + '/'
                print("dir_path:" + dir_path)
                files = os.listdir(dir_path)
                count = 0

                for file in files:
                    if 'dcm' not in file:
                        continue
                    count += 1
                    print(count)

                    # load original CT dicom file
                    file_path = dir_path + file
                    ds = pydicom.dcmread(file_path)

                    # compute index
                    org_idx = int(ds.InstanceNumber)
                    nii_idx = sct_data.shape[-1] + 1 - org_idx

                    # find the corresponding nii data and change the form
                    nii_pix = np.zeros((512, 512), dtype=np.int16)
                    if modality == 'CT':
                        nii_pix = sct_data[:, :, nii_idx - 1] + 1024
                    else:
                        nii_pix = sct_data[:, :, nii_idx - 1]
                    nii_pix = nii_pix.T
                    nii_pix = np.flip(nii_pix, 0)

                    for n, value in enumerate(nii_pix.flat):
                        ds.pixel_array.flat[n] = value

                    # save the new dicom data
                    ds.PixelData = ds.pixel_array.tostring()
                    # print(ds.pixel_array[255, 255])
                    new_path = dir_path.split("/" + modality)[0] + '/' + cs_file.split('.nii')[0]
                    print("new_path:" + new_path)
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    ds.save_as(new_path + '/' + file)


# Description: get the header info
def header(dcm_path):
    path = dcm_path
    ds = pydicom.dcmread(path)
    print(ds)


if __name__ == '__main__':
    nii_dcm_auto()