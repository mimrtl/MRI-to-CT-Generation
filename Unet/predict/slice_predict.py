import numpy as np
import os
from keras.models import load_model
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Description: make predictions with well-trained model for bmp file
def slice_predict(model, file_path, type):
    mr_file = Image.open(file_path)
    mr_data = np.array(mr_file)
    mr_data = mr_data / 255.0
    mr = np.zeros((1, 512, 512, 1))
    mr[0, :, :, 0] = mr_data

    y_hat = np.zeros((512, 512))
    y_hat[:, :] = np.squeeze(model.predict(mr))[:, :, 0]
    y_hat *= 255.0

    y_hat = Image.fromarray(y_hat)
    y_hat = y_hat.convert('L')

    out_dir = file_path.split('prep_4_sCT')[0] + '/' + type + '/sCT/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = out_dir + file_path.split('/')[-1].split('MR')[0] + 'sCT' + file_path.split('/')[-1].split('MR')[-1]

    y_hat.save(out_path, 'bmp')


if __name__ == '__main__':
    model_path = '/code/data/per/gp_1/10-cases/1st_round/gen/model/loss_model_chansey-2020-07-16-01-07.hdf5'
    model = load_model(model_path, compile=False)
    type = 'val'
    path = '/code/data/per/gp_1/10-cases/data_for_dis/prep_4_sCT/MR-' + type + '/'
    file_dir = os.listdir(path)
    for i in file_dir:
        print(i)
        file_path = path + i
        slice_predict(model, file_path, type)
