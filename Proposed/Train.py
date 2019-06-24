# -*- coding: utf-8 -*-
"""
QL
"""

from keras import backend as K
from keras import optimizers
from keras import callbacks
from keras.utils import plot_model
from keras.models import load_model
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import re
import tensorflow as tf

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


K.set_learning_phase(1) #set learning phase
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
import keras
print('The keras version is ',keras.__version__)
print('The TF version is ',tf.__version__)



# setting the hyper parameters
# import os
import argparse

# setting the hyper parameters
parser = argparse.ArgumentParser(description="Encoder")

parser.add_argument('--continueToTrainFlag', default=False, type=bool)
parser.add_argument('--featureFlag', default=0, type=int)  # featureFlag (0)
parser.add_argument('--outputFlag', default=1, type=int)  # outputFlag (0 mel, 1 linear)

parser.add_argument('--newSeedNum', default=418571351248, type=int)  # new seeds for shuffle data when continue training
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
parser.add_argument('--save_dir', default=None)
parser.add_argument('--is_training', default=1, type=int)
parser.add_argument('-w', '--weights', default=None, help="The path of the saved weights")
parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate")
parser.add_argument('--lr_decay', default=0.98, type=float, help="The value multiplied by lr at each epoch.")
args = parser.parse_args()
print(args)


############################################################
from DataGenerator import dataGenBig as dataGenBig
# dg = dataGenBig()

if args.continueToTrainFlag:
    dg = dataGenBig(seedNum=args.newSeedNum, verbose=False, verboseDebugTime=False)
    # dg = dataGenBig(seedNum=123456789, verbose=False, verboseDebugTime=False)
else:
    dg = dataGenBig(seedNum = 123456789, verbose = False, verboseDebugTime=False)
# [aa,bb] = dg.myDataGenerator(0)# change yield to return to debug the generator


from LossFuncs import my_loss as customLoss


if args.outputFlag==0:
    from GenerateModels import EncoderNetBigMel as GenerateModel
elif args.outputFlag==1:
    from GenerateModels import EncoderNetBigLinear as GenerateModel



tag = 'EncoderModel_{}_FF{}_CustomLoss'.format(['mel','linear'][args.outputFlag],args.featureFlag)

print(tag)
if args.save_dir is None:
    save_dir = './EncoderResult/Models/{}'.format(tag)
else:
    save_dir = args.save_dir


# Load the model
train_model = GenerateModel()
print(train_model.summary())
# plot_model(train_model, to_file='Model.png')



initial_epoch = 0
if args.continueToTrainFlag:
    try:
        savedModels = [filename for path, subdirs, files in os.walk(save_dir)
                          for filename in files if filename.endswith(".h5")]

        if len(savedModels)>0:
            index = []
            for saveModelName in savedModels:
                temp = re.findall(r'(?<=model-)(\d+).h5', saveModelName)
                if len(temp)>0:
                    index.append(int(temp[0]))
            # index = [int(re.findall(r'(?<=weights-)(\d+).h5', saveModelName)[0]) for saveModelName in savedModels
            #          if re.findall(r'(?<=weights-)(\d+).h5', saveModelName) is not None]
            if len(index)>0:
                initial_epoch = max(index)
            if initial_epoch>=1 and args.continueToTrainFlag:
                # since the optimiser status is not saved, save-weight only is not applicable for retraining
                # train_model.load_weights("{}/weights-{}.h5".format(save_dir,initial_epoch))
                del train_model
                # from keras.utils.generic_utils import get_custom_objects
                # get_custom_objects().update({"my_loss": customLoss})
                # train_model = load_model("{}/model-{:02d}.h5".format(save_dir, initial_epoch))
                # train_model = load_model("{}/model-{:02d}.h5".format(save_dir, initial_epoch),
                #                          custom_objects={'my_loss': customLoss})
                modelName = "{}/model-{:02d}.h5".format(save_dir, initial_epoch)
                train_model = load_model(modelName, custom_objects={'my_loss': customLoss})
                print('\nA pre-trained model {} has been loaded, continue training\n'.format(modelName))
                print(train_model.summary())
                # Initial_lr = Initial_lr*(lr_decay**initial_epoch)
            else:
                initial_epoch = 0
    except:
        print('\nCould not find a pre-trained model, start a refresh training\n')



if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# train_model.compile(loss={'kmeans_o': affinitykmeans}, optimizer=optimizers.Nadam(clipnorm=CLIPNORM))
# compile the model
if args.continueToTrainFlag and initial_epoch>=1:
    # will also take care of compiling the model using the saved training configuration
    # (unless the model was never compiled in the first place).
    pass
else:
    # Note the key words loss and losses are very different! made a stupid error here and wasted a whole day
    train_model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=[customLoss], metrics=[customLoss])
    # train_model.compile(optimizer=optimizers.Adam(lr=args.lr),
    #                     loss={'spectrum': customLoss, 'mask': 'mse'}, metrics={'spectrum': customLoss, 'mask': 'mse'})

log = callbacks.CSVLogger(save_dir + '/log.csv')
tb = callbacks.TensorBoard(log_dir=save_dir + '/tensorboard-logs',
                           batch_size=dg.BATCH_SIZE_Train, histogram_freq=args.debug)
checkpoint = callbacks.ModelCheckpoint(save_dir + '/model-{epoch:02d}.h5', monitor='val_loss',
                                       save_best_only=True, save_weights_only=False, verbose=1, period=10)
lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

train_model.fit_generator(generator=dg.myDataGenerator(dataFlag=0, featureFlag=args.featureFlag, outputFlag=args.outputFlag), # 0 training batch, 1 validation batch
                    steps_per_epoch=dg.EpochBatchNum[0],
                    epochs=args.epochs,
                    initial_epoch = initial_epoch,
                    validation_data=dg.myDataGenerator(dataFlag=1,featureFlag=args.featureFlag, outputFlag=args.outputFlag),
                    validation_steps=dg.EpochBatchNum[1],
                    callbacks=[log, tb, checkpoint, lr_decay])
# End: Training with data augmentation -----------------------------------------------------------------------#
#
# train_model.save_weights(save_dir + '/trained_model_weights.h5') # saves only the weight
# train_model.save(save_dir + '/trained_model.h5') # the weight, the architecture, the optimiser status, for re-training
# print('Trained model saved to \'%s/trained_model.h5\'' % save_dir)
#
#
# # If you wish to test source separation, generate a mixed 'mixed.wav'
# # file and test with the following line
# # separate_sources('mixed.wav', model, 2, 'out')


