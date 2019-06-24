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

# plt.figure(1)
# plt.xlabel("age")
# plt.ylabel("bbbb")
# plt.title("xxxxxx")
# plt.show()

import numpy as np
import scipy.io as sio
import scipy.signal as sg
import os
import sys
from random import shuffle
import random
import time
import re
from hparams import hparams
import audio
import datetime
import pickle
import librosa
from os.path import join

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial




# # Make a nested list based on the name
# def nestedListFromName(alist):
#     IDList = []
#     NestedList = []
#     for file in alist:
#         temp = os.path.basename(file)[:2]
#         if temp not in IDList:
#             IDList.append(temp)
#             NestedList.append([file])
#         else:
#             ind = IDList.index(temp)
#             NestedList[ind].append(file)
#
#     return NestedList

def ExtractFeatureFromOneSignal(featureFlag, outputFlag, s1, interf_path, s2_scale, chooseIndexNormalised, batchLen, saveNameMix=None):
    # if chooseIndexNormalised is None, we extract blocks of features from [0, batchLen, batchLen*2, batchLen*3 ... Length-batchLen]
    # if saveNameMix is None, don't save the mixture

    # First try to load a target (from LJSpeech) and a concatenated interference (from the same speaker in TSP)

    # orderFlag = 0 # 0: male target, female interference        1 female target, male interference
    try:

        # the target signal
        # s1 = np.load(target_path)
        # the interference signal
        s2_original, _ = librosa.load(interf_path) # both the target and the interference are sampled at 22050 Hz
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
        # sys.exit()
        currentSubBatchIn1 = None
        currentSubBatchIn2 = None
        currentSubBatchOut = None
        chooseIndex = None
        return (currentSubBatchIn1, currentSubBatchIn2, currentSubBatchOut, chooseIndex)



    # # the RIRs for the target and the interference respectively
    # # h1, h2 = RIR[azi1], RIR[azi2]
    # #
    # # lr1 = np.stack([sg.fftconvolve(s1, h1[0]), sg.fftconvolve(s1, h1[1])], axis=0)
    # # lr2 = np.stack([sg.fftconvolve(s2, h2[0]), sg.fftconvolve(s2, h2[1])], axis=0)
    # #
    # # lr1 = lr1[:, :len(s1)]
    # # lr2 = lr2[:, :len(s2)]
    #
    # # print the input SNR
    # inputSNR = 20*np.log10(np.linalg.norm(lr1)/np.linalg.norm(lr2))
    # # print("The input SNR is {} dB".format(inputSNR))

    # first normalise s2
    s2 = s2 * (s2_scale/max(abs(s2)))

    mixture = s1+s2

    # # print the input SNR
    # inputSNR = 20*np.log10(np.linalg.norm(s1)/np.linalg.norm(s2))
    # print("The input SNR is {} dB".format(inputSNR))

    if hparams.rescaling:
        scale = 1 / max(abs(mixture)) * hparams.rescaling_max
    else:
        scale = 1 / max(abs(mixture)) * 0.99  # normalise the mixture thus the maximum magnitude = 0.99

    mixture *= scale

    target = s1*scale

    interf = s2*scale


    # save
    if saveNameMix is not None:
        # dst_dir = "../mixtures"
        # os.makedirs(dst_dir, exist_ok=True)
        # print(time.strftime("%Y%m%d-%H%M%S"))
        # time.sleep(2)
        # time_suffix = time.strftime("%Y%m%d-%H%M%S")
        # dst_wav_path = join(dst_dir, "Ind_{}_mix.wav".format(time_suffix))
        # librosa.output.write_wav(dst_wav_path, mixture, sr=hparams.sample_rate)
        librosa.output.write_wav(saveNameMix, mixture, sr=hparams.sample_rate)



    # plt.figure(10)
    # plt.subplot(211)
    # plt.plot(lr[0,:100])
    # plt.plot(lr[1,:100],'r')
    # plt.show()




    # the linear spectrum in dB scale
    (mixLogPower, _) = audio.linearspectrogram(mixture)
    mixLogPower = mixLogPower[0:-1]
    # plt.figure(17)
    # temp = np.reshape(mixLogPower, (np.product(mixLogPower.shape),))
    # plt.hist(temp[:513*40], density=True, bins=100)
    # plt.ylabel('Probability')
    # plt.show()


    # the mel-scale mixture spectrum
    mix_mels = audio.melspectrogram(mixture[(hparams.fft_size - audio.get_hop_size()):]).astype(np.float32)


    # target_path_mel = target_path.replace('audio','mel')
    # mels_dry = np.load(target_path_mel)

    # the melscale spectrum in dB scale

    if outputFlag==0:
        target_mels = audio.melspectrogram(target[(hparams.fft_size - audio.get_hop_size()):]).astype(np.float32)
        interf_mels = audio.melspectrogram(interf[(hparams.fft_size - audio.get_hop_size()):]).astype(np.float32)

    elif outputFlag==1:# generate the mask labels
        (targetLogPower, _) = audio.linearspectrogram(target)
        targetLogPower = targetLogPower[0:-1]
        (interfLogPower, _) = audio.linearspectrogram(interf)
        interfLogPower = interfLogPower[0:-1]

        # plt.figure(10)
        # ax1 = plt.subplot(311)
        # im1 = ax1.pcolor(mix_mels[:, 0:200])
        # plt.colorbar(im1)
        # ax2 = plt.subplot(312, sharex=ax1)
        # im2 = ax2.pcolor(target_mels[:, 0:200])
        # plt.colorbar(im2)
        # ax3 = plt.subplot(313, sharex=ax1)
        # im3 = ax3.pcolor(interf_mels[:, 0:200])
        # plt.colorbar(im3)
        # plt.show()



    # plt.figure(10)
    # ax1 = plt.subplot(311)
    # im1 = ax1.pcolor(mixLogPower[:,0:200])
    # plt.colorbar(im1)
    #
    # ax2 = plt.subplot(312, sharex=ax1)
    # (temp, _) = audio.linearspectrogram(lr1[0])
    # im2 = ax2.pcolor(temp[:,0:200])
    # plt.colorbar(im2)
    #
    # ax3 = plt.subplot(313, sharex=ax1)
    # (temp, _) = audio.linearspectrogram(lr2[0])
    # im3 = ax3.pcolor(temp[:,0:200])
    # plt.colorbar(im3)
    # plt.show()




    # plt.figure(10)
    # plt.subplot(211)
    # temp = np.reshape(mels_dry, (np.product(mels_dry.shape),))
    # plt.hist(temp, normed=True, bins=30)
    # plt.ylabel('Probability')
    # plt.subplot(212)
    # temp = np.reshape(target_mels, (np.product(target_mels.shape),))
    # plt.hist(temp, normed=True, bins=30)
    # plt.ylabel('Probability')
    # plt.show()



    # do the normalisation and conversion for DNN input and output

    # doing thresholding to the audio energy then normalisation
    # currentThreshold = np.max(mixLogPower) - threshd_linSpec  # lower than x dB
    # mixLogPower[mixLogPower < currentThreshold] = currentThreshold
    # mixLogPower -= dataGenBigObj.ref_level_db
    # mixLogPower /= dataGenBigObj.ref_std
    # currentThreshold = np.max(target_mels) - threshd_melSpec
    # target_mels[target_mels < currentThreshold] = currentThreshold
    # target_mels -= dataGenBigObj.refmel_level_db
    # target_mels /= dataGenBigObj.refmel_std



    # incase a sequence is shorter than 100 frames
    try:
        # Based on the length of the target, we randomly extract different number of batches of samples
        # for instance, around 100 frames, 1 sample. around 1000 frames, 10 samples.
        # randomly choose batchLen=100 consecutive frames in each sequence

        # # If you really want to randomise the training use this one
        # N = int(round(mixLogPower.shape[1] / batchLen))
        # a = np.arange(mixLogPower.shape[1] - batchLen)
        # shuffle(a)
        # chooseIndex = a[:N]

        # If you want to repeat your results, use this one
        if chooseIndexNormalised is None:
            chooseIndex = np.arange(0, mixLogPower.shape[1], batchLen, dtype=int)
            chooseIndex[-1] = mixLogPower.shape[1] - batchLen
        else:
            chooseIndex = (chooseIndexNormalised * (mixLogPower.shape[1] - batchLen)).astype(int)

        N = len(chooseIndex)

        # concatenate the feature as the input
        Index1 = (np.tile(range(0, batchLen), (N, 1))).T
        Index2 = np.tile(chooseIndex, (batchLen, 1))
        Index = Index1 + Index2

        # DNN input
        mixLogPower = mixLogPower[:, Index]  # (W,T)--->(W,100,N)
        mix_mels = mix_mels[:, Index] # (W_mel,T)--->(W_mel,100,N)

        subBatchIn1 = mixLogPower.transpose([2, 1, 0])
        subBatchIn2 = mix_mels.transpose([2, 1, 0])

        # DNN output
        if outputFlag==0: # mel scale output
            target_mels_block = target_mels[:, Index]  # (W,T)--->(W,100,N) the target mel spectrum
            interf_mels_block = interf_mels[:, Index]
            # subBatchOut = target_mels_block.transpose([2, 1, 0])  # (N,100,W)

            # Get dominant spectra indexes, create one-hot outputs
            Y = np.zeros((target_mels.shape[0], Index.shape[0], Index.shape[1]) + (2,)).astype(np.float32)  # (W,100,N,2)

            specs = np.stack((target_mels_block, interf_mels_block), axis=-1)  # (W,100,N,2)
            vals = np.argmax(specs, axis=-1)  # (W,100,N)
            for i in range(2):
                t = np.zeros(2).astype(np.float32)
                t[i] = 1
                Y[vals == i] = t

            subBatchOut = np.concatenate((np.expand_dims(target_mels_block, axis=3),Y),axis=-1) # (W,100,N,3)
            subBatchOut = subBatchOut.transpose([2, 1, 0, 3])  # (W,100,N,3)---> (N,100,W,3)

        elif outputFlag==1: # linear scale output
            target_linear_block = targetLogPower[:, Index]  # (W,T)--->(W,100,N) the target linear spectrum
            interf_linear_block = interfLogPower[:, Index]

            # Get dominant spectra indexes, create one-hot outputs
            Y = np.zeros((targetLogPower.shape[0], Index.shape[0], Index.shape[1]) + (2,)).astype(np.float32) #(W,100,N,2)

            specs = np.stack((target_linear_block, interf_linear_block), axis=-1)#(W,100,N,2)
            vals = np.argmax(specs, axis=-1)#(W,100,N)
            for i in range(2):
                t = np.zeros(2).astype(np.float32)
                t[i] = 1
                Y[vals == i] = t

            # plt.figure(17)
            # temp = np.reshape(mixLogPower, (np.product(mixLogPower.shape),))
            # plt.hist(temp, normed=True, bins=50)
            # plt.ylabel('Probability')
            # plt.show()

            # # Create mask for zeroing out gradients from silence components
            # silenceThreshold = 0.08
            # z = np.zeros(2).astype(np.float32)
            # Y[mixLogPower < silenceThreshold] = z

            subBatchOut = np.concatenate((np.expand_dims(target_linear_block, axis=3), Y), axis=-1)  # (W,100,N,3)
            subBatchOut = subBatchOut.transpose([2, 1, 0, 3])  # (W,100,N, 3)---> (N,100,W,3)


        # plt.figure(17)
        # plt.subplot(411)
        # temp = np.reshape(mixLP_L, (np.product(mixLP_L.shape),))
        # plt.hist(temp, normed=True, bins=30)
        # plt.ylabel('Probability')
        # plt.subplot(412)
        # temp = np.reshape(mixLP_R, (np.product(mixLP_R.shape),))
        # plt.hist(temp, normed=True, bins=30)
        # plt.ylabel('Probability')
        # plt.subplot(413)
        # temp = np.reshape(mixLogPower, (np.product(mixLogPower.shape),))
        # plt.hist(temp, normed=True, bins=30)
        # plt.ylabel('Probability')
        # plt.subplot(414)
        # temp = np.reshape(target_mels, (np.product(target_mels.shape),))
        # plt.hist(temp, normed=True, bins=30)
        # plt.ylabel('Probability')
        # plt.show()

        # index = 4
        # plt.figure(100)
        # plt.title('aaa')
        # ax1 = plt.subplot(411)
        # im1 = ax1.pcolor(subBatchOut[index, :, :].squeeze().T)
        # plt.colorbar(im1)
        # ax2 = plt.subplot(412, sharex=ax1)
        # im2 = ax2.pcolor(subBatchIn[index, :, :,0].squeeze().T)
        # plt.colorbar(im2)
        # ax3 = plt.subplot(413, sharex=ax1)
        # im3 = ax3.pcolor(subBatchIn[index, :, :,1].squeeze().T)
        # plt.colorbar(im3)
        # ax4 = plt.subplot(414, sharex=ax1)
        # im4 = ax4.pcolor(subBatchIn[index, :, :, 2].squeeze().T)
        # plt.colorbar(im4)
        # plt.show()


        currentSubBatchIn1 = subBatchIn1[:N]
        currentSubBatchIn2 = subBatchIn2[:N]  # the mel spectrum
        currentSubBatchOut = subBatchOut[:N]
        # if i + N > batchSize:
        #     N = dataGenBigObj.EpochBatchNum[dataFlag] - i
        # BatchDataIn[i:i + N] = subBatchIn[:N]
        # BatchDataOut[i:i + N] = subBatchOut[:N]

        # i += N
    except:
        currentSubBatchIn1 = None
        currentSubBatchIn2 = None
        currentSubBatchOut = None
        chooseIndex = None


    return (currentSubBatchIn1, currentSubBatchIn2, currentSubBatchOut, chooseIndex)


class dataGenBig:

    def __init__(self, seedNum = 123456789, verbose = False, verboseDebugTime = True):
        # self.seedNum = seedNum
        self.verbose = verbose
        self.verboseDebugTime = verboseDebugTime

        num_workers = min(cpu_count()-2,6) # parallel at most 4 threads
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
        self.parallelN = 4

        self.BATCH_SIZE_Train = 32 #32 # 128 #4096  # mini-batch size
        # self.batchSeqN_Train = self.BATCH_SIZE_Train

        self.BATCH_SIZE_Valid = 32  # 32 # 128 #4096  # mini-batch size
        # self.batchSeqN_Valid = self.BATCH_SIZE_Valid

        # self.EpochBatchNum = [200,20]
        self.EpochBatchNum = [200, 20] # [100*32*100*256/20050 ~ 1.1 hour, ~15 minutes]

        self.batchLen = 64 # 2^N for ease of model design
        self.halfNFFT = int(hparams.fft_size/2)  # instead of nfft/2+1, we keep only 2^N for ease of model design

        self.target_train_i = 0
        self.target_valid_i = 0
        self.target_test_i = 0

        # # load the RIRs for mixture simulations
        # mat = sio.loadmat('./RIRs/G57_mic_2205.mat')
        # self.RIR = np.asarray(mat['ir'])
        # self.azis = mat['azis'].flatten()
        # # load the groundtruth IPDs for feature extraction
        # mat1 = sio.loadmat('./RIRs/G57_2205_IPDParams.mat')
        # self.IPD_mean = (np.asarray(mat1['IPD_mean']).T)[:,:-1]

        # The training and testing data. The pkl file is generated in AudioProc.py
        with open('./Data/EncoderTrainTestSignalList.pkl', 'rb') as f:
            target_train, target_test, interf_train, interf_test, _, _ = pickle.load(f)

        self.target_test = target_test
        self.interf_test = interf_test

        ################### divide the training set to train and valid 80-20
        self.trainNum = int(round(len(target_train) * 0.8))
        self.validNum = len(target_train) - self.trainNum
        self.target_train = target_train[:self.trainNum]
        self.target_valid = target_train[self.trainNum::]

        self.trainNum_interf = int(round(len(interf_train) * 0.8))
        self.validNum_interf = len(interf_train) - self.trainNum_interf
        self.interf_train = interf_train[:self.trainNum_interf]
        self.interf_valid = interf_train[self.trainNum_interf::]

        self.train_i = 0
        self.valid_i = 0

        random.seed(seedNum)
        shuffle(self.target_train)
        shuffle(self.target_valid)
        shuffle(self.interf_train)
        shuffle(self.interf_valid)

    # # When trying to train data at different sessions, we might need to shuffle the data with different order
    # def setNewSeed(self,seedNum = 888888):
    #     random.seed(self.seedNum)

    def myDataGenerator(self, dataFlag=0, featureFlag = 0, outputFlag = 0):
        # dataFlag 0 train, 1 valid, 2 test
        # featureFlag: Different input feature combinations (0,1)
        # outputFlag: Different output features 0 mel_spectrum 1 linear spectrum 2 IBM

        # plt.figure(1)
        # plt.xlabel("aaaa")
        # plt.ylabel("nnnn")
        # plt.title("xxxxx")
        # plt.show()

        outputFreNum = [hparams.num_mels, self.halfNFFT][outputFlag]
        batchSize = [self.BATCH_SIZE_Train, self.BATCH_SIZE_Valid][dataFlag]

        cout = 3 # The groundtruth spectrum (1 channel) and the IBM labels (2-channel)associated labels for DC
        BatchDataIn1 = np.zeros((batchSize, self.batchLen, self.halfNFFT), dtype='f')  # 100 x 512  linear-scale
        BatchDataIn2 = np.zeros((batchSize, self.batchLen, hparams.num_mels), dtype='f')  # 100 x 80 mel-scale
        # size (number_sample, self.batchLen, #Mels)  100 x Wm
        BatchDataOut = np.zeros((batchSize, self.batchLen, outputFreNum, cout), dtype='f')


        # save for the next batch
        BatchDataIn1Next = np.zeros((batchSize * 2, self.batchLen, self.halfNFFT), dtype='f')
        BatchDataIn2Next = np.zeros((batchSize * 2, self.batchLen, hparams.num_mels), dtype='f')
        BatchDataOutNext = np.zeros((batchSize * 2, self.batchLen, outputFreNum, cout), dtype='f')

        batchNum = 0
        availableN = 0 # number of unused samples generated from the previous round of parallel executor

        while 1:
            if self.verbose:
                print('\nNow collect a mini batch for {}'.format(['training','validataion'][dataFlag]))

            time_collect_start = datetime.datetime.now()
            NinCurrentBatch=0

            if availableN>0:
                tempAvailableN = min(availableN,batchSize)
                # print('\n Grab unused {} samples from the previou round of parallel processing'.format(tempAvailableN))
                BatchDataIn1[:tempAvailableN] = BatchDataIn1Next[:tempAvailableN]
                BatchDataIn2[:tempAvailableN] = BatchDataIn2Next[:tempAvailableN]
                BatchDataOut[:tempAvailableN] = BatchDataOutNext[:tempAvailableN]
                availableN = max(availableN-tempAvailableN,0)
                # There are too many unused samples from the previous round
                if availableN>0:
                    BatchDataIn1Next[:availableN] = BatchDataIn1Next[tempAvailableN:tempAvailableN+availableN]
                    BatchDataIn2Next[:availableN] = BatchDataIn2Next[tempAvailableN:tempAvailableN + availableN]
                    BatchDataOutNext[:availableN] = BatchDataOutNext[tempAvailableN:tempAvailableN + availableN]
                NinCurrentBatch += tempAvailableN

            while NinCurrentBatch<batchSize:


                futures = []

                # for each target sequence, randomly choose an interfering environmental signal and add them together

                sequence_i = [self.train_i,self.valid_i][dataFlag]

                for sequence_ii in range(sequence_i, sequence_i + self.parallelN):  # parallel 4 processes
                    # futures.append(self.executor.submit(partial(foo, dataFlag, sequence_ii)))

                    if dataFlag==0:
                        target_path = self.target_train[sequence_ii]
                        interf_id = random.randint(0, len(self.interf_train) - 1)
                        interf_path = self.interf_train[interf_id]
                    else:
                        target_path = self.target_valid[sequence_ii]
                        interf_id = random.randint(0, len(self.interf_valid) - 1)
                        interf_path = self.interf_valid[interf_id]


                    # print('\n ========={}======{}======={}========{}=======\n'.format(self.train_i, sequence_ii, target_path, interf_path))


                    # the target signal
                    target = np.load(target_path)
                    # get roughly the size of the target STFT
                    Nframe = int(len(target) / audio.get_hop_size())
                    Nsamples = int(round(Nframe / self.batchLen))
                    chooseIndexNormalised = np.asarray([random.random() for i in range(Nsamples)])  # extract N samples
                    # print(chooseIndexNormalised)

                    batchLen = self.batchLen

                    # interf_scale = random.random()*5
                    interf_scale = 0.5

                    # ExtractFeatureFromOneSignal(featureFlag, outputFlag, target, interf_path, interf_scale, chooseIndexNormalised, batchLen)

                    saveMixtureName = None
                    # saveMixtureName = "{}.wav".format(sequence_ii)

                    futures.append(self.executor.submit(
                        partial(ExtractFeatureFromOneSignal, featureFlag, outputFlag, target, interf_path, interf_scale,
                                chooseIndexNormalised, batchLen, saveMixtureName)))


                # [print(future.result()[0][0, 40, 30, 0]) for future in futures]
                tempResults = [future.result() for future in futures]

                # (currentSubBatchIn, currentSubBatchOut) = self.ExtractFeatureAssociatedOneTarget(dataFlag, sequence_i)

                for (currentSubBatchIn1, currentSubBatchIn2, currentSubBatchOut, _) in tempResults:

                    if currentSubBatchIn1 is not None:
                        N = len(currentSubBatchIn1)
                        if NinCurrentBatch + N > batchSize:
                            # these samples are not used in the current batch, we will save it for the next batch of data generation
                            N = batchSize - NinCurrentBatch
                            reuseableN = min(len(currentSubBatchIn1)+NinCurrentBatch-batchSize,batchSize*2-availableN)
                            BatchDataIn1Next[availableN:availableN + reuseableN] = currentSubBatchIn1[N:N+reuseableN]
                            BatchDataIn2Next[availableN:availableN + reuseableN] = currentSubBatchIn2[N:N + reuseableN]
                            BatchDataOutNext[availableN:availableN + reuseableN] = currentSubBatchOut[N:N+reuseableN]
                            availableN += reuseableN
                        if N>0:
                            BatchDataIn1[NinCurrentBatch:NinCurrentBatch + N] = currentSubBatchIn1[:N]
                            BatchDataIn2[NinCurrentBatch:NinCurrentBatch + N] = currentSubBatchIn2[:N]
                            BatchDataOut[NinCurrentBatch:NinCurrentBatch + N] = currentSubBatchOut[:N]
                            NinCurrentBatch += N


                # sequence_i += self.parallelN
                # print('\n ++++++++++++{}'.format(sequence_i))
                if dataFlag == 0:
                    self.train_i += self.parallelN
                    # if ((sequence_i >= self.trainNum-self.parallelN+1) | (self.train_i >=self.trainNum-self.parallelN+1)):
                    if (self.train_i >= self.trainNum - self.parallelN + 1):
                        self.train_i = 0
                        # sequence_i = 0
                        shuffle(self.target_train)
                elif dataFlag == 1:
                    self.valid_i += self.parallelN
                    if (self.valid_i >=self.validNum-self.parallelN+1):
                        self.valid_i = 0
                        shuffle(self.target_valid)

                # print('\n ++++++++++++{}++++{}+++++{}+++++++\n'.format(self.train_i, self.valid_i, sequence_i))


            time_collect_end = datetime.datetime.now()
            # print("\t The total time to collect the current batch of data is ", time_collect_end - time_collect_start)


            batchNum += 1
            # if batchNum > self.EpochBatchNum[dataFlag]:
            #     batchNum = 1
            if self.verbose:
                print('\n Batch {} data collected using time of '.format(batchNum), time_collect_end - time_collect_start, '\n')

            yield [BatchDataIn1, BatchDataIn2], [BatchDataOut]






if __name__=="__main__":


    # _mel_basis = audio._build_mel_basis()
    #
    # plt.figure(1)
    # plt.xlabel("aaaa")
    # plt.ylabel("nnnn")
    # plt.plot(_mel_basis[0::5].T)
    # plt.show()

    print('Test data generator')
    dg = dataGenBig(seedNum = 123456789, verbose = True, verboseDebugTime=False)

    # dg.myDataGenerator(0)
    # change yield to return to debug the generator
    featureFlag = 0
    outputFlag = 0
    [aa,aa2],[bb] = dg.myDataGenerator(0,featureFlag=featureFlag,outputFlag=outputFlag)
    # trainMode (0train 1validation 2test),
    # featureFlag (0 spectrum shift1 shift2, 1 spectrum spectrum*shift1 spectrum*shift2)
    # outputFlag (0mel, 1linear)
    # visualise the signal
    # import matplotlib.pyplot as plt
    #
    index = 31 #
    plt.figure(100)
    plt.title('Training input...')
    ax1 = plt.subplot(211)
    im1 = ax1.pcolor(aa[index,:,:].T)
    plt.colorbar(im1)
    ax2 = plt.subplot(212, sharex=ax1)
    im2 = ax2.pcolor(aa2[index,:,:].T)
    plt.colorbar(im2)

    # plt.show()

    plt.figure(101)
    plt.title('Groundtruth output')
    ax1 = plt.subplot(311)
    im1 = ax1.pcolor(bb[index, :, :, 0].T)
    plt.colorbar(im1)
    ax2 = plt.subplot(312, sharex=ax1)
    im2 = ax2.pcolor(bb[index, :, :, 1].T)
    plt.colorbar(im2)
    ax3 = plt.subplot(313, sharex=ax1)
    im3 = ax3.pcolor(bb[index, :, :, 2].T)
    plt.colorbar(im3)
    plt.show()
