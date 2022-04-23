import os
import shutil

import torchvision.transforms
from PIL import Image
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import soundfile
# from p_tqdm import p_map
from multiprocessing import Pool
import functools
from glob import glob

dir = '/mnt/data1/kwebst_data/models/saved_models/vgg19/001/img/epoch-40'
train_img_dir = f'/mnt/data1/kwebst_data/data/GOOD_MEL_IMAGES/fold1/train/'

def load_file(file):
    if file.endswith('.jpg'):
        i = Image.open(file)
        ic = torchvision.transforms.Resize(size=(100,100))(i)
        mat = np.array(ic)
        return [file, mat]


def get_dif(mat1, mat2):
    d = mat1 - mat2
    return np.sum(np.abs(d))


def load_train_imgs():
    mats = []
    names = []

    for root, subdirectories, files in os.walk(train_img_dir):
        fs = [os.path.join(root, file) for file in files]

        # p = Pool(12)
        # o = p.map(load_file, fs)
        o = [load_file(file) for file in fs]
        if len(o) > 0:
            mats += [n[1] for n in o]
            names += [n[0].replace('GOOD_MEL_IMAGES', 'GOOD_SOUNDS').replace('jpg', 'wav') for n in o]

    return mats, names


img_mats = []
img_names = []
def check_all_train_images(mat):
    global img_names, img_mats
    if len(img_mats) == 0:
        img_mats, img_names = load_train_imgs()

    f = functools.partial(get_dif, mat)
    # p = Pool(12)
    # diffs = p.map(f, img_mats)

    diffs = [f(m) for m in img_mats]

    idx = np.argmin(diffs)

    assert f(img_mats[idx]) == diffs[idx]

    return img_names[idx]



if __name__ == '__main__':
    num = 37

    files = glob(f"{dir}/prototype-img-original[0-9]*.png")

    for fi in files:
        i = Image.open(fi)
        original = np.array(i)

        closest_sound = check_all_train_images(np.average(original, axis=2))
        print(f'{fi} -> {closest_sound}')

        targ_file = fi.replace('img-original', 'sound').replace('png', 'wav')
        shutil.copyfile(closest_sound, targ_file)

    print('done')
    pass