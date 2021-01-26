import nibabel as nib
import numpy as np
import math
import os


# Description: compute mae, std, mse, psnr, and NCC
def compute_mae_psnr(gp_id):
    folder = '/project/data/nii_data/gp_' + str(gp_id) + '/test/'

    case_name = os.listdir(folder)
    total_case = []
    for case in case_name:
        print('---------' + case + '---------')
        temp_case = []
        temp_case.append(case)
        sCT_path = folder + case + '/sCT_30_2.nii'
        CT_path = folder + case + '/CT_target.nii'

        sCT_file = nib.load(sCT_path)
        CT_file = nib.load(CT_path)

        sCT_data = sCT_file.get_fdata()
        CT_data = CT_file.get_fdata()

        CT_data[CT_data < -1000] = -1000
        CT_data[CT_data > 2000] = 2000

        # CT_data = CT_data[:, :, 19:]
        # sCT_data = sCT_data[:, :, 19:]

        mae_loss = np.mean(np.abs(sCT_data-CT_data))
        print('mae loss:', mae_loss)
        temp_case.append(mae_loss)

        std_loss = np.std(np.abs(sCT_data-CT_data))
        print('std:', std_loss)
        temp_case.append(std_loss)

        mse_loss = np.mean(np.square(sCT_data-CT_data))
        print('mse loss:', mse_loss)
        temp_case.append(mse_loss)

        psnr = (10.0 * math.log((np.amax(CT_data) ** 2) / mse_loss)) / math.log(10)
        print('psnr:', psnr)
        temp_case.append(psnr)

        mean_CT = np.average(CT_data)
        mean_sCT = np.average(sCT_data)

        new_CT = CT_data - mean_CT
        new_sCT = sCT_data - mean_sCT

        numerator = np.sum(new_CT*new_sCT)
        denominator = np.sqrt(np.sum(new_CT*new_CT)*np.sum((new_sCT*new_sCT)))
        NCC = numerator / denominator

        print('NCC:', NCC)
        temp_case.append(NCC)
        print('\n')
        total_case.append(temp_case)
    return total_case


if __name__ == '__main__':
    total_group = []
    for i in range(1, 8):
        temp_group = []
        temp_group.append('group' + str(i))
        print('---------group' + str(i) + '---------')
        temp_group.append(compute_mae_psnr(i))
        print('\n')
        total_group.append(temp_group)
    print(total_group)
