import librosa as librosa
import numpy as np
import scipy
import soundfile as soundfile
import scipy.io

if __name__ == '__main__':
    num = 28
    signal, sr = librosa.load(f'../data/prototype-sound{num}.wav')

    # this is the number of samples in a window per fft
    n_fft = 2048
    print(sr)
    # The amount of samples we are shifting after each fft
    hop_length = 512

    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)

    im = np.interp(power_to_db, [-80, 0], [0, 255])  # map power_to_dp from [-80,0] to [0, 255]
    # im = np.rint(im)  # round to int
    im = im.astype('uint8')

    remap = np.interp(im, [0, 255], [-80, 0])
    remap = librosa.db_to_power(remap)
    audio = librosa.feature.inverse.mel_to_audio(remap, n_fft=int(n_fft), sr=int(sr), hop_length=int(hop_length))
    scipy.io.wavfile.write(f"./rebuild1-{num}.wav", sr, audio)
    audio = librosa.feature.inverse.mel_to_audio(remap, n_fft=int(n_fft), sr=int(sr * 2), hop_length=int(hop_length))
    scipy.io.wavfile.write(f"./rebuild2-{num}.wav", sr * 2, audio)
    audio = librosa.feature.inverse.mel_to_audio(remap, n_fft=int(n_fft), sr=int(sr / 2), hop_length=int(hop_length))
    scipy.io.wavfile.write(f"./rebuild3-{num}.wav", int(sr / 2), audio)

    pass