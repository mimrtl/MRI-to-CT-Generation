import os
import nibabel as nib


# Description: remove the slices outside the region of interest
def clip_data(path):
    case_dict = {'Case9': [25, 103], 'Case14': [45, 105], 'Case21': [0, 88], 'Case24': [0, 70],
                 'Case26': [15, 105]}
    in_path = path
    g = os.walk(in_path)
    for path, dir_list, file_list in g:
        if len(file_list) > 0:
            for file in file_list:
                if 'original' in file:
                    case_name = path.split('/')[-1]
                    range_z = case_dict[case_name]

                    nii_path = path + '/' + file
                    nii_file = nib.load(nii_path)
                    nii_data = nii_file.get_fdata()

                    new_data = nii_data[:, :, range_z[0]:range_z[1]]

                    affine = nii_file.affine
                    header = nii_file.header

                    new_file = nib.Nifti1Image(new_data, affine, header)
                    if 'MR' in file:
                        final_path = path + '/' + 'MR_clip.nii'
                    elif 'CT' in file:
                        final_path = path + '/' + 'CT_clip.nii'
                    else:
                        continue
                    print(final_path)
                    nib.save(new_file, final_path)
                else:
                    continue


if __name__ == '__main__':
    path = '/project/data/nii_data/gp_1/train/'
    clip_data(path)