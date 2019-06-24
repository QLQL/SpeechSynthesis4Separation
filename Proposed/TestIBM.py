# use IBM to generate the optimal BSS results
# The commented block code in the end generate results for Proposed-res-gt
import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
print("Using:",matplotlib.get_backend())

from keras import backend as K
import random
import numpy as np
import pickle
import argparse
import os
import librosa
import audio
from hparams import hparams

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


K.set_learning_phase(0) #set learning phase
K.set_image_data_format('channels_last')
# to check if there are gpu available
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

GPUFlag = True
ExistGPUs = get_available_gpus()
if len(ExistGPUs)==0:
    GPUFlag = False


############################################################
dst_dir = "./NewResults"
os.makedirs(dst_dir,exist_ok=True)
dst_dir_mel = os.path.join(dst_dir,'temp')
os.makedirs(dst_dir_mel,exist_ok=True)

############################################################
############################################################
############################################################
from DataGenerator import dataGenBig as dataGenBig, ExtractFeatureFromOneSignal
dg = dataGenBig(seedNum = 123456789, verbose = False, verboseDebugTime=False)
# [aa,bb] = dg.myDataGenerator(0)# change yield to return to debug the generator




import os
import librosa
from hparams import hparams


# This is for statistical analysis
SampleN = 100
mseResults = []
mseResults2 = []
random.seed(66666666)

# # The following is to make use the same sequence are being used
# SampleN = 500
# sequence_i_save = np.zeros((SampleN),dtype='int32')
# interf_i_save = np.zeros((SampleN),dtype='int32')
#
# for i in range(SampleN): # SampleN groups of mixtures and separated signals.
#     sequence_i = random.randint(1, len(dg.target_test) - 1)
#     interf_i = random.randint(0, len(dg.interf_test) - 1)
#
#     sequence_i_save[i] = sequence_i
#     interf_i_save[i] = interf_i
#
#     print(sequence_i_save[i], '\n', interf_i_save[i])
#
#     # s1 = np.load(dg.target_test[sequence_i])  # read in the target
#
# # Saving the objects:
# with open('TestSignalList500.pkl', 'wb') as f:  # Python 2 open(..., 'w') Python 3: open(..., 'wb')
#     pickle.dump([sequence_i_save, interf_i_save], f)


with open('TestSignalList500.pkl','rb') as f:  # Python 3: open(..., 'rb')
    sequence_i_save, interf_i_save = pickle.load(f)


_mel_basis = audio._build_mel_basis() # for mel-scale outp
_mel_basis = _mel_basis[:,:-1].T

for sample_i in range(SampleN):  # SampleN groups of mixtures and separated signals.
    print("Sample number {}".format(sample_i + 1))
    sequence_i = sequence_i_save[sample_i]
    interf_i = interf_i_save[sample_i]

    target_path = dg.target_test[sequence_i]
    interf_path = dg.interf_test[interf_i]
    print(target_path, '\n', interf_path)

    # generate the mixture and features
    # tempSaveName = 'Ind_{}_Mixture.wav'.format(i)
    chooseIndexNormalised = None

    s1 = np.load(target_path)  # read in the target
    saveNameMix = None

    # get the mixture
    try:

        # the target signal
        # s1 = np.load(target_path)
        # the interference signal
        s2_original, _ = librosa.load(
            interf_path)  # both the target and the interference are sampled at 22050 Hz
        L = len(s1)
        s2 = s2_original.copy()
        while len(s2) < L:
            s2 = np.concatenate((s2, s2_original), axis=0)

        s2 = s2[:L]

        if (s1 is None) | (s2 is None):
            print("Data loading fail")
            sys.exit()

    except:
        print('Check you code for loading the data!')
        print(interf_path)

    s2_scale = 0.5
    s2 = s2 * (s2_scale / max(abs(s2)))

    mixture = s1 + s2

    (S1_linear_spectrum, reconstruction_angle) = audio.linearspectrogram(s1)
    (S2_linear_spectrum,_) = audio.linearspectrogram(s2)

    # SdB = audio._denormalize_linear(S1_linear_spectrum) + hparams.ref_level_db_linear
    # # then the magnitude spectrum abs(stft(signal))
    # Sfull = audio._db_to_amp(SdB)
    #
    # S1_mel_spectrum = audio._linear_to_mel(Sfull)
    #
    # S1_mel_unwrap = audio._mel_to_linear(S1_mel_spectrum)
    #
    # # plt.figure(1)
    # # ax1 = plt.subplot(211)
    # # ax1.pcolor(audio._amp_to_db(Sfull[:,:100]))
    # # ax2 = plt.subplot(212)
    # # ax2.pcolor(audio._amp_to_db(S1_mel_unwrap[:,:100]))
    # # plt.show()
    #
    # resInfo = Sfull-S1_mel_unwrap
    #
    # dst_npy_path = "./Melnpy/Ind_{}_est_mel.npy".format(sample_i)
    # estimate_mel_spec = np.load(dst_npy_path) # normalised linear
    #
    # SdB_mel = audio._denormalize(estimate_mel_spec) + hparams.ref_level_db
    # Sfull_mel = audio._db_to_amp(SdB_mel.T)
    # S1_mel_unwrap_est = audio._mel_to_linear(Sfull_mel)
    #
    # S1_mel_unwrap_est_res = np.clip(S1_mel_unwrap_est + resInfo,0,a_max=None)


    # fig, axes = plt.subplots(nrows=2, ncols=1)
    # ax1 = axes.flat[0]
    # ax1.pcolor(audio._amp_to_db(Sfull[:, :100]))
    # ax2 = axes.flat[1]
    # im = ax2.pcolor(audio._amp_to_db(S1_mel_unwrap_est_res[:, :100]))
    # fig.colorbar(im, ax=axes.ravel().tolist())
    # plt.show()





    (mix_spectrum, _) = audio.linearspectrogram(mixture)

    # first get the db-scale spectrum dB(abs(stft(signal)))
    SdB = audio._denormalize_linear(mix_spectrum) + hparams.ref_level_db_linear
    # plt.figure(100)
    # plt.title('Spectrum of the estimated signal')
    # im1 = plt.pcolor(SdB.T)
    # plt.colorbar(im1)

    # then the magnitude spectrum abs(stft(signal))
    Sfull = audio._db_to_amp(SdB)
    # apply the IBM
    Sfull[S1_linear_spectrum<S2_linear_spectrum] = 0.0001*Sfull[S1_linear_spectrum<S2_linear_spectrum]


    proposal_spectrogram = Sfull * np.exp(1.0j * reconstruction_angle)
    # proposal_spectrogram = S1_mel_unwrap_est_res * np.exp(1.0j * reconstruction_angle)
    waveform = audio.istft_for_reconstruction(proposal_spectrogram.T, hparams.fft_size, hparams.hop_size)
    dst_wav_path = "{}/Ind_{}_est_IBM.wav".format(dst_dir, sample_i)

    # save
    librosa.output.write_wav(dst_wav_path, waveform, sr=hparams.sample_rate)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
