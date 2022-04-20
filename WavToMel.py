import os
from pathlib import Path
import matplotlib.pyplot as plt
import librosa, librosa.display
import numpy as np
from tqdm import tqdm

src_data_dir = "/mnt/data1/kwebst_data/data/"
# src_data_dir = "D:/data/"
dataset_dir = "NSYNTH"
target_dir = "NSYNTH_MEL_IMAGES"


def main():
    directory = src_data_dir + dataset_dir

    for root, subdirectories, files in os.walk(directory):
        for file in tqdm(files):
            if (".wav" in file) and not (file.startswith("._")):
                try:
                    wav2mel(os.path.join(root, file))
                except:
                    print("Error on: "+os.path.join(root, file))


def wav2mel(src_filepath: str):
    # this is the number of samples in a window per fft
    n_fft = 2048
    # The amount of samples we are shifting after each fft
    hop_length = 512

    filepath, filename = os.path.split(src_filepath)
    name, src_ext = os.path.splitext(filename)

    tar_ext = '.jpg'  # want to save as a jpg
    # directory path in target + filename + file extension
    target_dirpath = filepath.replace(dataset_dir, target_dir)
    target_filepath = target_dirpath + "/" + name + tar_ext

    Path(target_dirpath).mkdir(parents=True, exist_ok=True)  # make sure the file exists

    signal, sr = librosa.load(src_filepath)

    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(8, 7))
    librosa.display.specshow(power_to_db, sr=sr, cmap ='magma', hop_length = hop_length)
    # plt.show()
    plt.savefig(target_filepath)
    plt.close()


if __name__ == "__main__":
    main()
