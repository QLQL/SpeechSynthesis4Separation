# split the LP dataset (target) for training and testing, using the same split as used for Tacotron synthesiser training
# ####split the TSP dataset (interference) for training 90-10
# ####split the ESC dataset (environmental noise as interference) for training 90-10
import numpy as np
from os import walk
from os.path import join
from nnmnkwii.datasets import FileDataSource
from sklearn.model_selection import train_test_split
from hparams import hparams
from random import shuffle, seed

class _NPYDataSource(FileDataSource):
    def __init__(self, data_root, col, speaker_id=None,
                 train=True, test_size=0.05, test_num_samples=None, random_state=1234):
        self.data_root = data_root
        self.col = col
        self.lengths = []
        self.speaker_id = speaker_id
        self.multi_speaker = False
        self.speaker_ids = None
        self.train = train
        self.test_size = test_size
        self.test_num_samples = test_num_samples
        self.random_state = random_state

    def interest_indices(self, paths):
        indices = np.arange(len(paths))
        if self.test_size is None:
            test_size = self.test_num_samples / len(paths)
        else:
            test_size = self.test_size
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=self.random_state)
        return train_indices if self.train else test_indices

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4 or len(l) == 5
        self.multi_speaker = len(l) == 5
        self.lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[2]), lines))

        paths_relative = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), paths_relative))

        if self.multi_speaker:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
            self.speaker_ids = speaker_ids
            if self.speaker_id is not None:
                # Filter by speaker_id
                # using multi-speaker dataset as a single speaker dataset
                indices = np.array(speaker_ids) == self.speaker_id
                paths = list(np.array(paths)[indices])
                self.lengths = list(np.array(self.lengths)[indices])

                # Filter by train/tset
                indices = self.interest_indices(paths)
                paths = list(np.array(paths)[indices])
                self.lengths = list(np.array(self.lengths)[indices])

                # aha, need to cast numpy.int64 to int
                self.lengths = list(map(int, self.lengths))
                self.multi_speaker = False

                return paths

        # Filter by train/test
        indices = self.interest_indices(paths)
        paths = list(np.array(paths)[indices])
        lengths_np = list(np.array(self.lengths)[indices])
        self.lengths = list(map(int, lengths_np))

        if self.multi_speaker:
            speaker_ids_np = list(np.array(self.speaker_ids)[indices])
            self.speaker_ids = list(map(int, speaker_ids_np))
            assert len(paths) == len(self.speaker_ids)

        return paths

    def collect_features(self, path):
        return np.load(path)

class RawAudioDataSource(_NPYDataSource):
    def __init__(self, data_root, **kwargs):
        super(RawAudioDataSource, self).__init__(data_root, 0, **kwargs)

target_data_root = '/DoChangeYourDirectoryHere/LJSpeechProcess/'
speaker_id = None

# the same train-test split as used for WaveNet training
target_train = RawAudioDataSource(target_data_root, speaker_id=speaker_id,
                                  train=True,
                                  test_size=hparams.test_size,
                                  test_num_samples=hparams.test_num_samples,
                                  random_state=hparams.random_state).collect_files()

target_test = RawAudioDataSource(target_data_root, speaker_id=speaker_id,
                                  train=False,
                                  test_size=hparams.test_size,
                                  test_num_samples=hparams.test_num_samples,
                                  random_state=hparams.random_state).collect_files()

interf_data_root = '/DoChangeYourDirectoryHere/ESC-50-master/audio'
import pickle
with open('./Data/ESCsequenceList.pkl','rb') as f:  # Python 3: open(..., 'rb')
    audioList, categoryList = pickle.load(f)

interf = [join(interf_data_root,recording) for recording in audioList]

seed(hparams.random_state)
permuteIndex = np.arange(len(interf)).astype(int)
np.random.shuffle(permuteIndex)

# interf[ppermuteIndex] is not working here
interf = [interf[i] for i in permuteIndex]
categoryList = [categoryList[i] for i in permuteIndex]

trainN = int(round(len(interf) * 0.9))
interf_train = interf[:trainN]
interf_test = interf[trainN::]

interf_train_cat = categoryList[:trainN]
interf_test_cat = categoryList[trainN::]

import pickle
# Saving the objects:
with open('./Data/EncoderTrainTestSignalList.pkl', 'wb') as f:  # Python 2 open(..., 'w') Python 3: open(..., 'wb')
    pickle.dump([target_train, target_test, interf_train, interf_test, interf_train_cat, interf_test_cat], f)

# # Getting back the objects:
# with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
#     obj0, obj1, obj2 = pickle.load(f)
