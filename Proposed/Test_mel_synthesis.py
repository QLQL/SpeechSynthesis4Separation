
import os
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



####################
SampleN = 100
for sample_i in range(SampleN):
    # useing wavenet to synthesis time-domain signal
    # load the mel-spectrum
    dst_npy_path = "{}/temp/Ind_{}_est_mel.npy".format(dst_dir, sample_i)
    estimate_spec = np.load(dst_npy_path)

    # file_name = splitext(split(conditional_path)[1])[0]
    # time_suffix = time.strftime("%Y%m%d-%H%M%S")

    dst_wav_path = "{}/Ind_{}_est_mel.wav".format(dst_dir, sample_i)

    # DO generate
    waveform = wavegen(model, length, c=estimate_spec, g=speaker_id, initial_value=initial_value, fast=True)

    # save
    librosa.output.write_wav(dst_wav_path, waveform, sr=hparams.sample_rate)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))

