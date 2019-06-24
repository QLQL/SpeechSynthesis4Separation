

*** data preparation

**The experiment is applided to simulate noisy mixture from the LJ speech dataset (https://keithito.com/LJ-Speech-Dataset/) and ESC environmental noise dataset (https://github.com/karoldvl/ESC-50). Download these two data sets first

** Run ProcessESC.py to pre-process ESC dataset, note you need to change the directory to the downloaded ESC data in the code. 
./Data/ESCsequences.pkl will be generated.

** load some preset parameters ./mycheckpoint/20180510_mixture_lj_checkpoint_step000320000_ema.json, which is available at (https://github.com/r9y9/wavenet_vocoder)

** Run ProcessLJ.py to pre-process the LJ speech dataset, note you need to change the directory to the downloaded LJ speech data and directory to the processed data in the code. 
Some processed data will be saved under file folder /LJSpeechProcess

** Run AudioProc.py to split data



*** Train encoder

** Run Train.py, the --outputFlag decides using the proposed (Mel-spectrum output) model or B1 (linear spectrum output) model.
The trained model will be saved ./EncoderResult/Models

** You can either modify trainWaveNet.py to train the decoder model, or use a pre-trained decoder model. The trained model should be under ./mycheckpoint. This paper uses a pre-trained model at https://github.com/r9y9/wavenet_vocoder


*** Inference

** First run Test.py (with --outputFlag=0) to run the encoder and generate some npy files containing the mel-spectrum, then run Test_mel_synthesis.py to run the speech synthesis decoder. The reason they were split into two parts is that, due to my workstation memory constraints, I couldn't load both models in one go.

** For linear baseline method B1, run Test.py (with --outputFlag=1)


*** Evaluations

** The objective evaluations are implemented with Matlab, under ./RIRs



