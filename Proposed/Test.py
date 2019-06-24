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


# setting the hyper parameters
parser = argparse.ArgumentParser(description="TestEncoder")

parser.add_argument('--featureFlag', default=0, type=int)  #
parser.add_argument('--outputFlag', default=1, type=int)  # outputFlag (0 mel, 1 linear)
parser.add_argument('--synthesisFlag', default=1, type=int)  # Synthesis or not
parser.add_argument('--showMode', default=0, type=int)  # Synthesis or not

args = parser.parse_args()
print(args)

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


from keras.utils.generic_utils import get_custom_objects
from LossFuncs import my_loss as customLoss
get_custom_objects().update({"my_loss": customLoss})


if args.outputFlag==0:
    from GenerateModels import EncoderNetBigMel as GenerateModel
elif args.outputFlag==1:
    from GenerateModels import EncoderNetBigLinear as GenerateModel
tag = 'EncoderModel_{}_FF{}_CustomLoss'.format(['mel','linear'][args.outputFlag],args.featureFlag)

# Load the model
EncoderModel = GenerateModel()
# print(model.summary())
# plot_model(model, to_file='FrequencyModel.png')

modelDirectory = './EncoderResultNov/Models/{}'.format(tag)
# The separation model
from keras.models import load_model
EncoderModel = load_model("{}/ConvergeModel.h5".format(modelDirectory))

print(EncoderModel.summary())
# plot_model(train_model, to_file='Model.png')

if args.synthesisFlag & args.outputFlag==1:
    import os
    import librosa
    from hparams import hparams


# This is for statistical analysis
SampleN = 100
mseResults = []
mseResults2 = []
random.seed(66666666)

# # The following is to make sure the same sequence are being used
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


# # This is for one example
# SampleN = 1
# mseResults = []
# mseResults2 = []
# temp = 66
# sequence_i_save = [sequence_i_save[temp]]
# interf_i_save = [interf_i_save[temp]]


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

    # # save
    # dst_wav_path = "{}/Ind_{}_source.wav".format(dst_dir, sample_i)
    # librosa.output.write_wav(dst_wav_path, s1, sr=22050)
    # saveNameMix = "{}/Ind_{}_mixture.wav".format(dst_dir, sample_i)

    (DNN_in1, DNN_in2, groundtruth, chooseIndex) = ExtractFeatureFromOneSignal(args.featureFlag, args.outputFlag, s1,
                                                                               interf_path, s2_scale=0.5,
                                                                               chooseIndexNormalised=chooseIndexNormalised,
                                                                               batchLen=dg.batchLen, saveNameMix=saveNameMix)

    DNN_in = [DNN_in1, DNN_in2]
    # to estimate the spectrum directly, either in mel (0) or linear scale (1)
    # apply the source separation
    estimate = EncoderModel.predict(DNN_in)

    Dim = [80, 512][args.outputFlag]
    L = chooseIndex[-1] + dg.batchLen
    estimate_spec = np.zeros((L, Dim))
    groundtruth_spec = np.zeros((L, Dim))
    mix_spec_mel = np.zeros((L, 80))
    mix_spec_linear = np.zeros((L, 512))


    for n, i in enumerate(chooseIndex):
        # the mixture
        mix_spec_linear[i:i + dg.batchLen] = DNN_in1[n]
        mix_spec_mel[i:i + dg.batchLen] = DNN_in2[n]
        # the groundtruth
        groundtruth_spec[i:i + dg.batchLen] = groundtruth[n,:,:,0]
        # the estimation
        estimate_spec[i:i + dg.batchLen] = estimate[n,:,:,0]

    # import scipy.io as sio
    # sio.savemat('./PlotResult.mat', dict([('mix_spec_linear', mix_spec_linear),('mix_spec_mel', mix_spec_mel), ('groundtruth_spec', groundtruth_spec), ('estimate_spec', estimate_spec)]))

    if args.showMode:
        plt.figure(100)
        plt.title('Mixtures, groundtruth and estimations')
        ax1 = plt.subplot(411)
        im1 = ax1.pcolor(mix_spec_mel[:100].T)
        plt.colorbar(im1)
        ax2 = plt.subplot(412, sharex=ax1)
        im2 = ax2.pcolor(mix_spec_linear[:100].T)
        plt.colorbar(im2)
        ax3 = plt.subplot(413, sharex=ax1)
        im3 = ax3.pcolor(groundtruth_spec[:100].T)
        plt.colorbar(im3)
        ax4 = plt.subplot(414, sharex=ax1)
        im4 = ax4.pcolor(estimate_spec[:100].T)
        plt.colorbar(im4)
        plt.show()

    if args.synthesisFlag:

        if args.outputFlag == 0:
            # useing wavenet to synthesis time-domain signal
            c = estimate_spec

            dst_npy_path = "{}/temp/Ind_{}_est_mel.npy".format(dst_dir, sample_i)
            # np.save(dst_npy_path,estimate_spec)

            print("Finished! Please run WaveNet to mel-spectrum {} to synthesize audio samples.".format(dst_dir))

        if args.outputFlag == 1:

            # first get the db-scale spectrum dB(abs(stft(signal)))
            SdB = audio._denormalize_linear(estimate_spec) + hparams.ref_level_db_linear
            # plt.figure(100)
            # plt.title('Spectrum of the estimated signal')
            # im1 = plt.pcolor(SdB.T)
            # plt.colorbar(im1)

            # then the magnitude spectrum abs(stft(signal))
            S = audio._db_to_amp(SdB)
            Sfull = np.concatenate((S,np.zeros(shape=(S.shape[0],1))),axis=-1)

            GriffinMode = 0
            if GriffinMode:
                # using the grinffin-lim algorithm to reconstruct time-domain signal
                # The grinffin-lim algorithm
                waveform = audio.reconstruct_signal_griffin_lim(Sfull, hparams.fft_size, hparams.hop_size, 100)

                dst_wav_path = "{}/Ind_{}_est_linear.wav".format(dst_dir, sample_i)

            else:

                # the direct overlap-and-add using phase info from the mixture/grountruth
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

                # (_, reconstruction_angle) = audio.linearspectrogram(mixture)
                (_, reconstruction_angle) = audio.linearspectrogram(s1)
                proposal_spectrogram = Sfull * np.exp(1.0j * reconstruction_angle.T)
                waveform = audio.istft_for_reconstruction(proposal_spectrogram, hparams.fft_size, hparams.hop_size)
                dst_wav_path = "{}/Ind_{}_est_linear2.wav".format(dst_dir, sample_i)


            # save
            librosa.output.write_wav(dst_wav_path, waveform, sr=hparams.sample_rate)

            print("Finished! Check out {} for generated audio samples.".format(dst_dir))

    # save one example



    # if args.outputFlag==1:
    #     # convert from linear scale to mel-scale
    #     SdB = audio._denormalize_linear(groundtruth_spec) + hparams.ref_level_db_linear
    #     D = audio._db_to_amp(SdB) # the STFT magnitude
    #     S = audio._amp_to_db(np.dot(D,_mel_basis)) - hparams.ref_level_db
    #     groundtruth_spec = audio._normalize(S)
    #
    #     SdB = audio._denormalize_linear(estimate_spec) + hparams.ref_level_db_linear
    #     D = audio._db_to_amp(SdB)  # the STFT magnitude
    #     S = audio._amp_to_db(np.dot(D, _mel_basis)) - hparams.ref_level_db
    #     estimate_spec = audio._normalize(S)


    # calculate the loss functions
    mseloss = (np.linalg.norm(groundtruth_spec - estimate_spec)) ** 2 / np.prod(estimate_spec.shape)
    refloss = (np.linalg.norm(groundtruth_spec)) ** 2 / np.prod(estimate_spec.shape)
    print(mseloss, refloss)
    temploss = mseloss / refloss

    mseResults.append(temploss)
    print('\n the normalised average error is {}'.format(sum(mseResults) / len(mseResults)))

    diff = (groundtruth_spec - estimate_spec) ** 2
    weight1 = groundtruth_spec ** 2
    weight2 = estimate_spec ** 2
    weight = weight1 + (1 - weight1) * weight2
    weight_sum = np.sum(weight)
    weight_square_diff_sum = np.sum(diff * weight)
    temploss2 = weight_square_diff_sum / weight_sum

    mseResults2.append(temploss2)
    print('\n the weighted normalised average error is {}'.format(sum(mseResults2) / len(mseResults2)))
    print('\n the current mse and weighted errors are respectively {} and {}'.format(temploss, temploss2))
