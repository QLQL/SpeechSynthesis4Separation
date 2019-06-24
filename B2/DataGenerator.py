# by QL

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


from collections import namedtuple
MyStruct = namedtuple("MyStruct", "rescaling rescaling_max multiProcFlag")
hparams = MyStruct(rescaling=True, rescaling_max=0.999, multiProcFlag = False)

def ExtractFeatureFromOneSignal(s1, interf_path, s2_scale, input_length, samples_of_interest_indices, chooseIndexNormalised, batchLen, saveNameMix=None):
    # if chooseIndexNormalised is None, we extract blocks of features from [0, batchLen, batchLen*2, batchLen*3 ... Length-batchLen]
    # if saveNameMix is None, don't save the mixture


    # First try to load a target (from LJSpeech) and a concatenated interference (from the same speaker in TSP)
    # orderFlag = 0 # 0: male target, female interference        1 female target, male interference
    # try:

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

    # except:
    #     print('Check you code for loading the data!')
    #     print(interf_path)
    #     # sys.exit()
    #     currentSubBatchIn1 = None
    #     currentSubBatchIn2 = None
    #     currentSubBatchOut1 = None
    #     currentSubBatchOut2 = None
    #     chooseIndex = None
    #     return (currentSubBatchIn1, currentSubBatchIn2, currentSubBatchOut1, currentSubBatchOut2, chooseIndex)

    if L<input_length:
        print("Signal too short")
        currentSubBatchIn1 = None
        currentSubBatchIn2 = None
        currentSubBatchOut1 = None
        currentSubBatchOut2 = None
        chooseIndex = None
        return (currentSubBatchIn1, currentSubBatchIn2, currentSubBatchOut1, currentSubBatchOut2, chooseIndex)




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


    # try:
    # Based on the length of the target, we randomly extract different number of batches of samples
    # for instance, around 7739 frames, 1 sample. around 77390 frames, 10 samples.
    # randomly choose batchLen=100 consecutive frames in each sequence



    # If you want to repeat your results, use this one
    if chooseIndexNormalised is None:
        chooseIndex = np.arange(0, L-input_length, batchLen, dtype=int)
        chooseIndex[-1] = L-input_length
    else:
        chooseIndex = (chooseIndexNormalised * (L-input_length)).astype(int)

    N = len(chooseIndex)

    # concatenate the feature as the input
    Index1 = (np.tile(range(0, input_length), (N, 1)))
    Index2 = np.tile(chooseIndex, (input_length, 1)).T
    Index = Index1 + Index2

    # DNN input
    subBatchIn1 = mixture[Index]  # (T)--->(N,input_length)
    subBatchIn2 = np.zeros(shape=(N,5)) # This condition input is forced to equal [0 0 0 0 0]

    # DNN output
    IndexNew = Index[:,samples_of_interest_indices[0]:samples_of_interest_indices[1]]

    subBatchOut1 = target[IndexNew]
    subBatchOut2 = interf[IndexNew]

    # plt.figure(100)
    # plt.title('Training data..')
    # index = 5
    # ax1 = plt.subplot(311)
    # ax1.plot(subBatchIn1[index])
    # # ax1.autoscale(enable=True, axis='x', tight=True)
    # ax2 = plt.subplot(312)
    # ax2.plot(subBatchOut1[index])
    # # ax2.autoscale(enable=True, axis='x', tight=True)
    # ax3 = plt.subplot(313)
    # ax3.plot(subBatchOut2[index])
    # # ax3.autoscale(enable=True, axis='x', tight=True)

    # except:
    #     subBatchIn1 = None
    #     subBatchIn2 = None
    #     subBatchOut1 = None
    #     subBatchOut2 = None
    #     chooseIndex = None


    return (subBatchIn1, subBatchIn2, subBatchOut1, subBatchOut2, chooseIndex)


class dataGenBig:

    def __init__(self, model, seedNum = 123456789, verbose = False, verboseDebugTime = True):
        # self.seedNum = seedNum
        self.model = model
        self.verbose = verbose
        self.verboseDebugTime = verboseDebugTime

        num_workers = min(cpu_count()-2,6) # parallel at most 4 threads
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
        self.parallelN = 4

        self.BATCH_SIZE_Train = 10 #32 # 128 #4096  # mini-batch size
        # self.batchSeqN_Train = self.BATCH_SIZE_Train

        self.BATCH_SIZE_Valid = 10  # 32 # 128 #4096  # mini-batch size
        # self.batchSeqN_Valid = self.BATCH_SIZE_Valid

        # self.EpochBatchNum = [200,20]
        self.EpochBatchNum = [200, 20] # [100*32*100*256/20050 ~ 1.1 hour, ~15 minutes]

        self.batchLen = 64 # 2^N for ease of model design

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

    def myDataGenerator(self, dataFlag=0):
        # dataFlag 0 train, 1 valid, 2 test

        # plt.figure(1)
        # plt.xlabel("aaaa")
        # plt.ylabel("nnnn")
        # plt.title("xxxxx")
        # plt.show()

        batchSize = [self.BATCH_SIZE_Train, self.BATCH_SIZE_Valid][dataFlag]

        samples_of_interest_indices = self.model.samples_of_interest_indices
        samples_of_interest_indices = [samples_of_interest_indices[0],samples_of_interest_indices[-1]+1]

        target_field_length = self.model.target_field_length

        input_length1 = self.model.input_length
        input_length2 = 5 # force all num_condition = [0 0 0 0 0]
        BatchDataIn1 = np.zeros((batchSize, input_length1), dtype='f')
        BatchDataIn2 = np.zeros((batchSize, input_length2), dtype='f')
        # size (number_sample, self.batchLen, #Mels)  100 x Wm
        output_length = self.model.padded_target_field_length
        BatchDataOut1 = np.zeros((batchSize, output_length), dtype='f')
        BatchDataOut2 = np.zeros((batchSize, output_length), dtype='f')


        # save for the next batch
        BatchDataIn1Next = np.zeros((batchSize * 2, input_length1), dtype='f')
        BatchDataIn2Next = np.zeros((batchSize * 2, input_length2), dtype='f')
        BatchDataOut1Next = np.zeros((batchSize * 2, output_length), dtype='f')
        BatchDataOut2Next = np.zeros((batchSize * 2, output_length), dtype='f')

        batchNum = 0
        availableN = 0 # number of unused samples generated from the previous round of parallel executor

        while True:
            if self.verbose:
                print('\nNow collect a mini batch for {}'.format(['training','validataion'][dataFlag]))

            time_collect_start = datetime.datetime.now()
            NinCurrentBatch=0

            if availableN>0:
                tempAvailableN = min(availableN,batchSize)
                # print('\n Grab unused {} samples from the previou round of parallel processing'.format(tempAvailableN))
                BatchDataIn1[:tempAvailableN] = BatchDataIn1Next[:tempAvailableN]
                BatchDataIn2[:tempAvailableN] = BatchDataIn2Next[:tempAvailableN]
                BatchDataOut1[:tempAvailableN] = BatchDataOut1Next[:tempAvailableN]
                BatchDataOut2[:tempAvailableN] = BatchDataOut2Next[:tempAvailableN]
                availableN = max(availableN-tempAvailableN,0)
                # There are too many unused samples from the previous round
                if availableN>0:
                    BatchDataIn1Next[:availableN] = BatchDataIn1Next[tempAvailableN:tempAvailableN+availableN]
                    BatchDataIn2Next[:availableN] = BatchDataIn2Next[tempAvailableN:tempAvailableN + availableN]
                    BatchDataOut1Next[:availableN] = BatchDataOut1Next[tempAvailableN:tempAvailableN + availableN]
                    BatchDataOut2Next[:availableN] = BatchDataOut2Next[tempAvailableN:tempAvailableN + availableN]
                NinCurrentBatch += tempAvailableN

            while NinCurrentBatch<batchSize:


                futures = []
                tempResults = []
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


                    # the target signal whose sampling rate is sample_rate = 22050
                    target = np.load(target_path)
                    # get roughly the size
                    Nsamples = min(int(len(target)/(input_length1*2)),4) # from one sequence, collect at most 4 samples
                    chooseIndexNormalised = np.asarray([random.random() for i in range(Nsamples)])  # extract N samples
                    # print(chooseIndexNormalised)



                    # interf_scale = random.random()*5
                    interf_scale = 0.5

                    saveMixtureName = None
                    # saveMixtureName = "{}.wav".format(sequence_ii)

                    # ExtractFeatureFromOneSignal(target, interf_path, interf_scale, input_length1,
                    #                             samples_of_interest_indices,
                    #                             chooseIndexNormalised, target_field_length, saveMixtureName)

                    if hparams.multiProcFlag:
                        futures.append(self.executor.submit(
                            partial(ExtractFeatureFromOneSignal, target, interf_path, interf_scale, input_length1,
                                    samples_of_interest_indices, chooseIndexNormalised, target_field_length,
                                    saveMixtureName)))

                    else:
                        tempResult = ExtractFeatureFromOneSignal(target, interf_path, interf_scale, input_length1,
                                                    samples_of_interest_indices, chooseIndexNormalised,
                                                    target_field_length, saveMixtureName)
                        tempResults.append(tempResult)


                # [print(future.result()[0][0, 40, 30, 0]) for future in futures]
                if hparams.multiProcFlag:
                    tempResults = [future.result() for future in futures]


                for (currentSubBatchIn1, currentSubBatchIn2, currentSubBatchOut1, currentSubBatchOut2, _) in tempResults:

                    if currentSubBatchIn1 is not None:
                        N = len(currentSubBatchIn1)
                        if NinCurrentBatch + N > batchSize:
                            # these samples are not used in the current batch, we will save it for the next batch of data generation
                            N = batchSize - NinCurrentBatch
                            reuseableN = min(len(currentSubBatchIn1)+NinCurrentBatch-batchSize,batchSize*2-availableN)
                            BatchDataIn1Next[availableN:availableN + reuseableN] = currentSubBatchIn1[N:N+reuseableN]
                            BatchDataIn2Next[availableN:availableN + reuseableN] = currentSubBatchIn2[N:N + reuseableN]
                            BatchDataOut1Next[availableN:availableN + reuseableN] = currentSubBatchOut1[N:N+reuseableN]
                            BatchDataOut2Next[availableN:availableN + reuseableN] = currentSubBatchOut2[
                                                                                    N:N + reuseableN]
                            availableN += reuseableN
                        if N>0:
                            BatchDataIn1[NinCurrentBatch:NinCurrentBatch + N] = currentSubBatchIn1[:N]
                            BatchDataIn2[NinCurrentBatch:NinCurrentBatch + N] = currentSubBatchIn2[:N]
                            BatchDataOut1[NinCurrentBatch:NinCurrentBatch + N] = currentSubBatchOut1[:N]
                            BatchDataOut2[NinCurrentBatch:NinCurrentBatch + N] = currentSubBatchOut2[:N]
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

            batch = {'data_input': BatchDataIn1, 'condition_input': BatchDataIn2}, {
                'data_output_1': BatchDataOut1, 'data_output_2': BatchDataOut2}

            yield batch

            # yield [BatchDataIn1, BatchDataIn2], [BatchDataOut1,BatchDataOut1]






if __name__=="__main__":

    print('Test data generator')

    import json
    import logging

    def load_config(config_filepath):
        try:
            config_file = open(config_filepath, 'r')
        except IOError:
            logging.error('No readable config file at path: ' + config_filepath)
            exit()
        else:
            with config_file:
                return json.load(config_file)

    config = load_config('./config.json')

    import models

    MC = models.DenoisingWavenet(config=config)


    print('Test data generator')
    dg = dataGenBig(MC, seedNum = 123456789, verbose = True, verboseDebugTime=False)


    # dg.myDataGenerator(0) # comment out the yield/return for debug
    # change yield to return to debug the generator
    Abatch = dg.myDataGenerator(0)
    # trainMode (0train 1validation 2test),
    # featureFlag (0 spectrum shift1 shift2, 1 spectrum spectrum*shift1 spectrum*shift2)
    # outputFlag (0mel, 1linear)
    # visualise the signal
    # import matplotlib.pyplot as plt
    #
    index = 12 #
    plt.figure(100)
    plt.title('Training input...')
    plt.plot(Abatch[0]['data_input'][index])

    # plt.show()

    plt.figure(101)
    plt.title('Groundtruth output')
    ax1 = plt.subplot(211)
    ax1.plot(Abatch[1]['data_output_1'][index])
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(Abatch[1]['data_output_2'][index])
    plt.show()









