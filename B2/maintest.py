# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Main.py
# Modified at April 2019 by QL, for test only

import sys
import logging
import optparse
import json
import os
import models
import util
import denoise
import numpy as np
import random
import librosa



def set_system_settings():
    sys.setrecursionlimit(50000)
    logging.getLogger().setLevel(logging.INFO)


def get_command_line_arguments():
    parser = optparse.OptionParser()
    parser.set_defaults(config='config1.json')
    parser.set_defaults(mode='training')
    parser.set_defaults(load_checkpoint=None)
    parser.set_defaults(condition_value=0)
    parser.set_defaults(batch_size=None)
    parser.set_defaults(one_shot=False)
    parser.set_defaults(clean_input_path=None)
    parser.set_defaults(noisy_input_path=None)
    parser.set_defaults(print_model_summary=False)
    parser.set_defaults(target_field_length=None)

    parser.add_option('--mode', dest='mode')
    parser.add_option('--print_model_summary', dest='print_model_summary')
    parser.add_option('--config', dest='config')
    parser.add_option('--load_checkpoint', dest='load_checkpoint')
    parser.add_option('--condition_value', dest='condition_value')
    parser.add_option('--batch_size', dest='batch_size')
    parser.add_option('--one_shot', dest='one_shot')
    parser.add_option('--noisy_input_path', dest='noisy_input_path')
    parser.add_option('--clean_input_path', dest='clean_input_path')
    parser.add_option('--target_field_length', dest='target_field_length')


    (options, args) = parser.parse_args()

    return options


def load_config(config_filepath):
    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        logging.error('No readable config file at path: ' + config_filepath)
        exit()
    else:
        with config_file:
            return json.load(config_file)


def get_valid_output_folder_path(outputs_folder_path):
    j = 1
    while True:
        output_folder_name = 'samples_%d' % j
        output_folder_path = os.path.join(outputs_folder_path, output_folder_name)
        if not os.path.isdir(output_folder_path):
            os.mkdir(output_folder_path)
            break
        j += 1
    return output_folder_path


def inference(config, cla):
    from collections import namedtuple
    MyStruct = namedtuple("MyStruct", "rescaling rescaling_max multiProcFlag")
    hparams = MyStruct(rescaling=True, rescaling_max=0.999, multiProcFlag=False)

    outputfolder = 'bbbbbb'
    os.makedirs(outputfolder, exist_ok=True)

    import pickle
    with open('Data/TestSignalList500.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        sequence_i_save, interf_i_save = pickle.load(f)

    # This is for statistical analysis
    SampleN = 100
    random.seed(66666666)

    # # This is for one example
    # SampleN = 1
    # temp = 66
    # sequence_i_save = [sequence_i_save[temp]]
    # interf_i_save = [interf_i_save[temp]]

    # Instantiate Model
    model = models.DenoisingWavenet(config, load_checkpoint=cla.load_checkpoint,
                                    print_model_summary=cla.print_model_summary)
    model.model.load_weights(cla.load_checkpoint)



    from DataGenerator import dataGenBig
    dg = dataGenBig(model, seedNum=123456789, verbose=False)
    s2_scale = 0.5

    for sample_i in range(SampleN):  # SampleN groups of mixtures and separated signals.
        print("Sample number {}".format(sample_i + 1))
        sequence_i = sequence_i_save[sample_i]
        interf_i = interf_i_save[sample_i]

        target_path = dg.target_test[sequence_i]
        interf_path = dg.interf_test[interf_i]
        print(target_path, '\n', interf_path)

        # generate the mixture

        s1 = np.load(target_path)  # read in the target
        s2_original, _ = librosa.load(interf_path)  # both the target and the interference are sampled at 22050 Hz
        L = len(s1)
        s2 = s2_original.copy()
        while len(s2) < L:
            s2 = np.concatenate((s2, s2_original), axis=0)

        s2 = s2[:L]

        if (s1 is None) | (s2 is None):
            print("Data loading fail")
            sys.exit()

        # first normalise s2
        s2 = s2 * (s2_scale / max(abs(s2)))

        mixture = s1 + s2
        if hparams.rescaling:
            scale = 1 / max(abs(mixture)) * hparams.rescaling_max
        else:
            scale = 1 / max(abs(mixture)) * 0.99  # normalise the mixture thus the maximum magnitude = 0.99

        mixture *= scale


        input = {'noisy':mixture,'clean':None}

        print("Denoising: " + target_path.split('/')[-1])
        batch_size = 10
        if config['model']['condition_encoding'] == 'one_hot':
            condition_input = util.one_hot_encode(int(cla.condition_value), 29)[0]
        else:
            condition_input = util.binary_encode(int(cla.condition_value), 29)[0]

        dst_wav_name = "Ind_{}_est_wavenet_".format(sample_i)
        denoise.denoise_sample(model, input, condition_input, batch_size, dst_wav_name, 22050, outputfolder)



if __name__ == "__main__":

    # set_system_settings()
    cla = get_command_line_arguments()
    cla.load_checkpoint = 'sessionsNew/0011/checkpoints/checkpoint.hdf5'

    config = load_config(cla.config)

    inference(config, cla)

    a = 0