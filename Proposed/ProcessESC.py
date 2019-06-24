# This Programe process ESC-50 (Dataset for Environmental Sound Classification)
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

audioDir = '/DoChangeYourDirectoryHere/ESC-50-master'
import librosa

import pandas as pd
import os
import numpy as np

def findFirstNone0index(array):
    # for idx, val in np.ndenumerate(array): # for multi-dimenstional data
    for idx in range(len(array)):
        if array[idx] != 0:
            return idx

def findLastNone0index(array):
    # for idx, val in np.ndenumerate(array): # for multi-dimenstional data
    for idx in range(len(array)):
        if array[-idx-1] != 0:
            return len(array)-idx-1


# Know nothing about pandas, so I am processing the table in a silly way
metaData = pd.read_csv(os.path.join(audioDir,'meta/esc50.csv')).sort_values('target')

recordings = metaData.ix[:, 0].tolist()
target = metaData.ix[:, 2].tolist()
category = metaData.ix[:, 3].tolist()

use = []
audioList = []
categoryList = []
for i in range(len(recordings)):
    #0 Animals
    # 1Natural soundscapes & water sounds
    # 2Human, non-speech sounds
    # 3Interior/domestic sounds
    # 4Exterior/urban noises

    if target[i] in np.append(np.arange(10,20),np.arange(40,50)): ##np.arange(40,50):#
        y, sr = librosa.load(os.path.join(audioDir,'audio',recordings[i]))

        energyBegining = np.sum(y[:100]**2)
        energyEnd = np.sum(y[-100:] ** 2)

        index1 = findFirstNone0index(y)
        index2 = findLastNone0index(y)

        timelength = (index2-index1)/sr

        if timelength>4.5:
            audioList.append(recordings[i])
            categoryList.append(category[i])
            print('Audio sig {} of type {}'.format(recordings[i],category[i]))



        # print('sr====={}'.format(sr))
        # print('{}------{} for sequence {} of type {}'.format(i, timelength, recordings[i], category[i]))

    # if energyBegining==0 or energyEnd==0:
    #     print('{} and {} for sequence {} of type {}'.format(energyBegining, energyEnd, recordings[i], category[i]))

    # plt.figure(1)
    # plt.plot(y)
    # plt.title("xxxxxx")
    # plt.show()

# Saving the objects:
import pickle
with open('./Data/ESCsequences.pkl', 'wb') as f:  # Python 2 open(..., 'w') Python 3: open(..., 'wb')
    pickle.dump([audioList, categoryList], f)
