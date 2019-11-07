# SpeechSynthesis4Separation

[__A Speech Synthesis Approach for High Quality Speech Separation and Generation__](https://github.com/QLQL/SpeechSynthesis4Separation/blob/master/SynthesisForSeparation.pdf)

__Qingju Liu, Philip JB Jackson, and Wenwu Wang__

We propose a new method for source separation by synthesizing the source from a speech mixture corrupted by various environmental noise. Unlike traditional source separation methods which estimate the source from the mixture as a replica of the original source (e.g. by solving an inverse problem), our proposed method is a synthesis-based approach which aims to generate a new signal (i.e. "fake" source) that sounds similar to the original source. The proposed system has an encoder-decoder topology, where the encoder predicts intermediate-level features from the mixture, i.e. Mel-spectrum of the target source, using a hybrid recurrent and hourglass network, while the decoder is a state-of-the-art WaveNet speech synthesis network conditioned on the Mel-spectrum, which directly generates time-domain samples of the sources. Both objective and subjective evaluations were performed on the synthesized sources, and show great advantages of our proposed method for high-quality speech source separation and generation.


Usage
-----
The folder /Proposed contains codes for the proposed method as well as the baseline method B1. The folder /B2 contains the baseline method B2. Most of the codes are supported by Keras and TensorFlow, the WaveNet part is supported by Pytorch.

Some audio samples (5*6) are audio provided under the folder /AudioSamples to demonstrate the separation performance with difference algorithms, with the clean dry groundtruth (_source), the noisy mixture (_mixture), sound estimated with the proposed method (_proposed), with the B1 baseline (_B1) and B2 baseline (_B2). The compressed folder of these audio samples is also uploaded in AudioSamples.zip.

**_First go to the folder /Proposed to run the proposed method and the baseline method B1_**
*****************************************************************************************

## Data preparation

The experiment is applided to noisy mixtures simulated from the [LJ speech dataset](https://keithito.com/LJ-Speech-Dataset/) and [ESC environmental noise dataset](https://github.com/karoldvl/ESC-50). Download these two data sets first.

Run ProcessESC.py to pre-process ESC dataset, note you need to change the directory to the downloaded ESC data in the code. 
./Data/ESCsequences.pkl will be generated.

load some preset parameters ./mycheckpoint/20180510_mixture_lj_checkpoint_step000320000_ema.json, which is available at the [link](https://github.com/r9y9/wavenet_vocoder)

Run ProcessLJ.py to pre-process the LJ speech dataset, note you need to change the directory to the downloaded LJ speech data and directory to the processed data in the code. 
Some processed data will be saved under file folder /LJSpeechProcess

Run AudioProc.py to split data. 
Some result is saved to ./Data/ESCsequenceList.pkl



## Train the encoder+decoder network

Run Train.py, the --outputFlag decides using the proposed (Mel-spectrum output) model or B1 (linear spectrum output) model.
The trained model will be saved ./EncoderResult/Models

You can either modify trainWaveNet.py to train the decoder model, or use a pre-trained decoder model. The trained model should be under ./mycheckpoint. This paper uses a pre-trained model at the [link](https://github.com/r9y9/wavenet_vocoder)


## Inference

First run Test.py (with --outputFlag=0) to run the encoder and generate some npy files containing the mel-spectrum, then run Test_mel_synthesis.py to run the speech synthesis decoder. The reason they were split into two parts is that, due to my workstation memory constraints, I couldn't load both models in one go.

For the linear baseline method B1, run Test.py (with --outputFlag=1)


## Evaluations

The objective evaluations are implemented with Matlab, under ./RIRs




**_Then go to the folder /B1 to run the baseline method B2_**
*****************************************************************************************

B1 is modified from the [method](https://github.com/drethage/speech-denoising-wavenet), to adapt to the tensorflow backend and our data set. Some contemporary results from folder ../Proposed need to be copied here such as ../Proposed/Data/ESCsequenceList.pkl

## Train

Run maintrain.py
Note that, if you don't have enough space to save all the checkpoints for every epoch, you might want to slightly change callbacks in models.py, e.g. save every few epochs or save only improving results or rewrite the previous saved model. 

## Inference
Run maintest.py, the path to the converged models, i.e. load_checkpoint needs to point to the saved converged model. 

