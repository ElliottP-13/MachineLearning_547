import json
import os
import random

import pydub

import WavConverter


def make_json_dataset(root, target):
    l = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            # print(os.path.join(path, name))
            filename = path + '/' + name
            classlabel = os.path.basename(os.path.dirname(filename))  # i hate this
            l.append({'wav': filename, 'labels': classlabel})
    with open(target, 'w') as f:
        json.dump({'data': l}, f)


def merge_audio(wavs, out='temp.wav', overlap=0.2):
    curr_wav = pydub.AudioSegment.from_wav(wavs[0])
    for i in range(1, len(wavs)):
        append = pydub.AudioSegment.from_wav(wavs[i])

        t = int(overlap * len(append))
        curr_wav = curr_wav.overlay(append, position=len(curr_wav) - t)  # first t bit of next gets mixed in
        curr_wav += append[t:]  # remaining bit plays alone

    curr_wav.export(out, format='wav')


def build_dset(n, dset_dir, json_dict):
    data = json.load(json_dict)['data']
    os.makedirs(dset_dir, exist_ok=True)

    wav_dir = dset_dir + '-wav'
    os.makedirs(wav_dir, exist_ok=True)

    for i in range(n):
        k = random.randint(1, 4)
        selected = random.sample(data, k)

        wavs = [s['wav'] for s in selected]
        labs = [s['labels'] for s in selected]

        wp = os.path.join(wav_dir, f"{n}-{'_'.join(labs)}.wav")
        ip = os.path.join(dset_dir, f"{n}-{'_'.join(labs)}")

        merge_audio(wavs, out=wp, overlap=0.2)
        img = WavConverter.convertWav(wp, target_filepath=ip, save=False)
        img.save(ip + '.jpg')

        # split up image into smaller ones
        w, h = img.size
        left = 0
        width = 244  # default size for protopnet
        count = 0
        while left < img.width:
            right = min(left + width, w)
            c = img.crop((left, 0, right, h))
            left += int(width * (1 - 0.2))
            c.save(f'{ip}-{count}.jpg')
            count += 1



if __name__ == '__main__':
    # root_path = "/mnt/data1/kwebst_data/data/GOOD_SOUNDS/fold1/train/"
    # make_json_dataset(root_path, "/mnt/data1/kwebst_data/data/GOOD_SOUNDS/fold1/train/train.json")
    json_file_path = "/mnt/data1/kwebst_data/data/GOOD_SOUNDS/fold1/train/train.json"
    dset_dir = '/mnt/data1/kwebst_data/data/GOOD_SYNTHETIC'
    build_dset(5000, dset_dir, json_file_path)
    pass