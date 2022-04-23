import os

import torchvision.transforms
from PIL import Image
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy.io.wavfile
import soundfile
# from p_tqdm import p_map
from multiprocessing import Pool
import functools

dir = '/mnt/data1/kwebst_data/models/saved_models/vgg19/001/img/epoch-40'


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
    f = 1
    train_img_dir = f'/mnt/data1/kwebst_data/data/GOOD_MEL_IMAGES/fold{f}/train/'

    mats = []
    names = []

    for root, subdirectories, files in os.walk(train_img_dir):
        fs = [os.path.join(root, file) for file in files]

        p = Pool(12)
        o = p.map(load_file, fs)
        if len(o) > 0:
            mats += [n[1] for n in o]
            names += [n[0].replace('GOOD_MEL_IMAGES', 'GOOD_SOUNDS') for n in o]

    return mats, names


img_mats = []
img_names = []
def check_all_train_images(mat):
    global img_names, img_mats
    if len(img_mats) == 0:
        img_mats, img_names = load_train_imgs()

    f = functools.partial(get_dif, mat)
    p = Pool(12)
    diffs = p.map(f, img_mats)

    idx = np.argmin(diffs)

    assert f(img_mats[idx]) == diffs[idx]

    return img_names[idx]



if __name__ == '__main__':
    num = 37
    plt.figure()  # zoomed in prototype
    one = img.imread(f'{dir}/prototype-img{num}.png')
    plt.imshow(one)
    plt.title('one')
    plt.show()

    plt.figure()  # base one
    two = img.imread(f'{dir}/prototype-img-original{num}.png')
    plt.imshow(two)
    plt.title('two')
    plt.show()

    plt.figure()  # heatmap
    three = img.imread(f'{dir}/prototype-img-original_with_self_act{num}.png')
    plt.imshow(three)
    plt.title('three')
    plt.show()

    npy = np.load(f'{dir}/prototype-self-act{num}.npy')
    print(npy)  # not sure what this is

    four = three - two
    five = np.interp(four, [np.amin(four), np.amax(four)], [0, 1])  # map power_to_dp from [-80,0] to [0, 255]

    activation_map = np.average(five, axis=2)  # doesn't work

    closest_sound = check_all_train_images(np.average(two, axis=2))
    print(closest_sound)



    # remap = np.interp(two, [0, 255], [-80, 0])
    # audio = librosa.feature.inverse.mel_to_audio(remap, n_fft=2048, sr=22050, hop_length=512)
    # scipy.io.wavfile.write(f"./prototype-{num}.wav", 22050, audio)
    # scipy.io.wavfile.write(f"./prototype3-{num}.wav", 22050, np.transpose(audio))
    # soundfile.write(f"./prototype4-{num}.wav", np.transpose(audio), 22050)
    # soundfile.write(f"./prototype2-{num}.wav", audio, 22050)

    print('done')
    pass