#  generate the wav files from groundtruth Mel spectrum
import os
import pickle
dst_dir = "./NewResults"
os.makedirs(dst_dir, exist_ok=True)

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

############################################################
from mysynthesis import *


checkpoint_path = "./mycheckpoint/20180510_mixture_lj_checkpoint_step000320000_ema.pth"
preset = "./mycheckpoint/20180510_mixture_lj_checkpoint_step000320000_ema.json"
speaker_id = None

length = 32000
initial_value = None
initial_value = None if initial_value is None else float(initial_value)
# From https://github.com/Rayhane-mamah/Tacotron-2
symmetric_mels = False
max_abs_value = -1

file_name_suffix = ""
output_html = False
speaker_id = None if speaker_id is None else int(speaker_id)
hparams2 = ""

# Load preset if specified
if preset is not None:
    with open(preset) as f:
        hparams.parse_json(f.read())

# Override hyper parameters
hparams.parse(hparams2)
assert hparams.name == "wavenet_vocoder"

from trainWaveNet import build_model

# Model
model = build_model().to(device)

# Load checkpoint
print("Load checkpoint from {}".format(checkpoint_path))
if use_cuda:
    checkpoint = torch.load(checkpoint_path)
else:
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint["state_dict"])
checkpoint_name = splitext(basename(checkpoint_path))[0]


with open('TestSignalList500.pkl','rb') as f:  # Python 3: open(..., 'rb')
    sequence_i_save, interf_i_save = pickle.load(f)


############################################################
from DataGenerator import dataGenBig as dataGenBig, ExtractFeatureFromOneSignal
dg = dataGenBig(seedNum = 123456789, verbose = False, verboseDebugTime=False)
# [aa,bb] = dg.myDataGenerator(0)# change yield to return to debug the generator



####################
SampleN = 100
for sample_i in range(50,SampleN):  # SampleN groups of mixtures and separated signals.
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


    # S1_mel_spectrum = audio.melspectrogram(s1)
    S1_mel_spectrum = audio.melspectrogram(s1[(hparams.fft_size - audio.get_hop_size()):]).astype(np.float32) # to be consistent with other methods

    S1_mel_spectrum = S1_mel_spectrum.T
    # DO generate
    waveform = wavegen(model, length, c=S1_mel_spectrum, g=speaker_id, initial_value=initial_value, fast=True)

    dst_wav_path = "{}/Ind_{}_est_gtWaveNet.wav".format(dst_dir, sample_i)

    # save
    librosa.output.write_wav(dst_wav_path, waveform, sr=hparams.sample_rate)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
