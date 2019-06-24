# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Main.py
# Modified at April 2019 by QL, for training only

import sys
import logging
import optparse
import json
import models


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

import keras
import os
def get_callbacks(model):

    return [
        keras.callbacks.ReduceLROnPlateau(patience=model.config['training']['early_stopping_patience'] / 2,
                                          cooldown=model.config['training']['early_stopping_patience'] / 4,
                                          verbose=1),
        keras.callbacks.ModelCheckpoint(os.path.join(model.checkpoints_path, 'checkpoint.hdf5'),save_best_only=True),
        keras.callbacks.CSVLogger(os.path.join(model.config['training']['path'], model.history_filename), append=True)
    ]

def training(config, cla):

    # Instantiate Model
    model = models.DenoisingWavenet(config, load_checkpoint=cla.load_checkpoint, print_model_summary=cla.print_model_summary)

    from DataGenerator import dataGenBig
    dg = dataGenBig(model, seedNum=123456789, verbose = False)

    model.model.fit_generator(
        generator=dg.myDataGenerator(dataFlag=0),
        # 0 training batch, 1 validation batch
        steps_per_epoch=config['training']['num_train_samples'],
        epochs=config['training']['num_epochs'],
        initial_epoch=0,
        validation_data=dg.myDataGenerator(dataFlag=1),
        validation_steps=config['training']['num_test_samples'],
        callbacks=get_callbacks(model)) # num_train_samples is actually num_train_batches


if __name__ == "__main__":
    # main()

    # set_system_settings()
    cla = get_command_line_arguments()
    cla.mode = 'training'
    cla.print_model_summary = True
    config = load_config(cla.config)
    training(config, cla)


    a = 0