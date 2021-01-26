import os
from nipype.interfaces.dcm2nii import Dcm2niix


# Description: convert Dicom files to Nifti file
def dcm_nii():
    g = os.walk("/project/data/dicom/gp_1/")
    for path, dir_list, file_list in g:
        print('*'*10)
        print(path)
        print(dir_list)
        print(file_list)
        if len(dir_list) == 0 and len(file_list) > 0:
            print(path)

            # print(len(file_list))
            converter = Dcm2niix()
            converter.inputs.bids_format = False
            converter.inputs.compress = 'n'
            converter.inputs.merge_imgs = True
            print(path)
            if path.split('/')[-1] == 'CT':
                temp = 'CT'
            else:
                temp = 'MR'
            converter.inputs.out_filename = temp
            outpath = path.split('New')[0] + 'New_nii' + path.split('New')[-1]
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            converter.inputs.output_dir = outpath
            converter.inputs.source_dir = path
            converter.run()

