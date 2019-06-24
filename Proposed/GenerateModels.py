from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Input, Add, MaxPool2D, Activation, Concatenate
from keras import layers, models, optimizers, initializers
from keras import backend as K
from keras.regularizers import l2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# import keras
# print("\n The keras version is ",keras.__version__)
# print("\n The TF version is ",tf.__version__)
K.set_learning_phase(1) #set learning phase
K.set_image_data_format('channels_last')

nModules = 1
def convBlock(x, numOut):

    SC = models.Sequential()
    SC.add(BatchNormalization())
    SC.add(Activation(activation='relu'))
    SC.add(Conv2D(filters=int(numOut / 2), kernel_size=(3, 3), padding='same'))
    SC.add(BatchNormalization())
    SC.add(Activation(activation='relu'))
    SC.add(Conv2D(filters=int(numOut / 2), kernel_size=(3, 3), padding='same'))
    SC.add(BatchNormalization())
    SC.add(Activation(activation='relu'))
    SC.add(Conv2D(filters=numOut, kernel_size=(1, 1), padding='same'))
    r = SC(x)

    # r = BatchNormalization()(x)
    # r = Activation(activation='relu')(r)
    # r = Conv2D(int(numOut / 2), (1, 1), padding='same')(r)
    # r = BatchNormalization()(r)
    # r = Activation(activation='relu')(r)
    # r = Conv2D(int(numOut / 2), (3, 3), padding='same')(r)
    # r = BatchNormalization()(r)
    # r = Activation(activation='relu')(r)
    # r = Conv2D(numOut, (1, 1), padding='same')(r)

    return r

def skipLayer(x, numOut):
    numIn = x.shape[3]
    if numIn == numOut:
        return x
    else:
        return Conv2D(numOut, (1,1), padding='same')(x)

def Residual(inp, numOut):
    c = convBlock(inp, numOut)
    s = skipLayer(inp, numOut)
    return Add()([c, s])

def hourglass(n, f, inp):
    # Upper branch
    up1 = inp
    for i in range(nModules):
        up1 = Residual(up1, f)

    # Lower branch
    low1 = MaxPool2D((2, 2), strides=2)(inp)
    for i in range(nModules):
        low1 = Residual(low1, f)

    low2 = None
    if n > 1:
        low2 = hourglass(n - 1, f, low1)
        low3 = low2
        for i in range(nModules):
            low3 = Residual(low3, f)
    else:
        low2 = low1
        for i in range(nModules):
            low2 = Residual(low2, f)
        low3 = low2



    up2 = UpSampling2D(size=(2, 2))(low3)

    # Bring two branches together
    return Add()([up1, up2])


#######################################################
#######################################################
#######################################################
def EncoderNetBigMel():
    num_freq = 512
    num_mels = 80
    num_frame = 64
    input_shape1 = (num_frame, num_freq)
    input_shape2 = (num_frame, num_mels)
    # output_shape = (num_frame, num_mels)

    input1 = layers.Input(shape=input_shape1, name='input1')
    input2 = layers.Input(shape=input_shape2, name='input2')

    L2R = 1e-7  # L2 regularization factor

    DROPOUT = 0.25  # Feed forward dropout
    RDROPOUT = 0.25  # Recurrent dropout


    input1_r1 = layers.Bidirectional(layers.LSTM(400, return_sequences=True,
                                                 kernel_regularizer=l2(L2R),
                                                 recurrent_regularizer=l2(L2R),
                                                 bias_regularizer=l2(L2R),
                                                 dropout=DROPOUT,
                                                 recurrent_dropout=RDROPOUT),name='linear_rnn')(input1)

    input1_td = layers.TimeDistributed(layers.Dense(4 * num_mels,
                                                   activation='linear',
                                                   kernel_regularizer=l2(L2R),
                                                   bias_regularizer=l2(L2R)),
                                                   name='linear_td')(input1_r1)

    input2_r1 = layers.Bidirectional(layers.LSTM(200, return_sequences=True,
                                                 kernel_regularizer=l2(L2R),
                                                 recurrent_regularizer=l2(L2R),
                                                 bias_regularizer=l2(L2R),
                                                 dropout=DROPOUT,
                                                 recurrent_dropout=RDROPOUT),name='mel_rnn')(input2)

    input2_td = layers.TimeDistributed(layers.Dense(4 * num_mels,
                                                    activation='linear',
                                                    kernel_regularizer=l2(L2R),
                                                    bias_regularizer=l2(L2R)),
                                                    name='mel_td')(input2_r1)

    def reshapeforCNN(inp):
        from keras import layers
        # inp_shape = (time, channels)
        target_shape = (inp._keras_shape[1], num_mels, int(inp._keras_shape[2] / num_mels))
        inputR = layers.Reshape(target_shape=target_shape)(inp)  # (100, 80, #)
        return inputR

    input1_td_R = layers.Lambda(reshapeforCNN)(input1_td)
    input2_td_R = layers.Lambda(reshapeforCNN)(input2_td)
    input2_R = layers.Lambda(reshapeforCNN)(input2)

    def concatenateFeature(inpList):
        x = K.concatenate(inpList, axis=-1)
        return x

    all_feature = layers.Lambda(concatenateFeature)([input1_td_R, input2_td_R, input2_R])
    hg_output = hourglass(3, 64, all_feature)

    hg_output_mel = layers.Lambda(concatenateFeature)([hg_output, input2_R])

    hg_cnn_output = Residual(hg_output_mel, 64)

    # hg_cnn_output2 = Residual(hg_cnn_output, 32)

    output_mel = layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),
                                       padding='same',activation='sigmoid')(hg_cnn_output)

    # # To remove the last channel with dimension of 1
    # def squeezeLastChannel(inp):
    #     x = K.squeeze(inp, axis=-1)
    #     return x
    #
    # output_mel_squeeze = layers.Lambda(squeezeLastChannel)(output_mel)
    #
    # BSS_model = models.Model(inputs=[input1,input2], outputs=output_mel_squeeze)

    BSS_model = models.Model(inputs=[input1, input2], outputs=output_mel)

    return BSS_model

def EncoderNetBigLinear():
    num_freq = 512
    num_mels = 80
    num_frame = 64
    input_shape1 = (num_frame, num_freq)
    input_shape2 = (num_frame, num_mels)
    # output_shape = (num_frame, num_mels)

    input1 = layers.Input(shape=input_shape1, name='input1')
    input2 = layers.Input(shape=input_shape2, name='input2')

    L2R = 1e-7  # L2 regularization factor

    DROPOUT = 0.25  # Feed forward dropout
    RDROPOUT = 0.25  # Recurrent dropout


    input1_r1 = layers.Bidirectional(layers.LSTM(400, return_sequences=True,
                                                 kernel_regularizer=l2(L2R),
                                                 recurrent_regularizer=l2(L2R),
                                                 bias_regularizer=l2(L2R),
                                                 dropout=DROPOUT,
                                                 recurrent_dropout=RDROPOUT),name='linear_rnn')(input1)

    input1_td = layers.TimeDistributed(layers.Dense(2 * num_freq,
                                                   activation='linear',
                                                   kernel_regularizer=l2(L2R),
                                                   bias_regularizer=l2(L2R)),
                                                   name='linear_td')(input1_r1)

    input2_r1 = layers.Bidirectional(layers.LSTM(200, return_sequences=True,
                                                 kernel_regularizer=l2(L2R),
                                                 recurrent_regularizer=l2(L2R),
                                                 bias_regularizer=l2(L2R),
                                                 dropout=DROPOUT,
                                                 recurrent_dropout=RDROPOUT),name='mel_rnn')(input2)

    input2_td = layers.TimeDistributed(layers.Dense(1 * num_freq,
                                                    activation='linear',
                                                    kernel_regularizer=l2(L2R),
                                                    bias_regularizer=l2(L2R)),
                                                    name='mel_td')(input2_r1)

    def reshapeforCNN(inp):
        from keras import layers
        # inp_shape = (time, channels)
        target_shape = (inp._keras_shape[1], num_freq, int(inp._keras_shape[2] / num_freq))
        inputR = layers.Reshape(target_shape=target_shape)(inp)  # (100, 80, #)
        return inputR

    input1_td_R = layers.Lambda(reshapeforCNN)(input1_td)
    input2_td_R = layers.Lambda(reshapeforCNN)(input2_td)
    input1_R = layers.Lambda(reshapeforCNN)(input1)

    def concatenateFeature(inpList):
        x = K.concatenate(inpList, axis=-1)
        return x

    all_feature = layers.Lambda(concatenateFeature)([input1_td_R, input2_td_R, input1_R])
    hg_output = hourglass(3, 64, all_feature)

    hg_output_linear = layers.Lambda(concatenateFeature)([hg_output, input1_R])

    hg_cnn_output = Residual(hg_output_linear, 64)

    # hg_cnn_output2 = Residual(hg_cnn_output, 32)

    output_linear = layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),
                                       padding='same',activation='sigmoid')(hg_cnn_output)

    # # To remove the last channel with dimension of 1
    # def squeezeLastChannel(inp):
    #     x = K.squeeze(inp, axis=-1)
    #     return x
    #
    # output_linear_squeeze = layers.Lambda(squeezeLastChannel)(output_linear)
    #
    # BSS_model = models.Model(inputs=[input1,input2], outputs=output_linear_squeeze)

    BSS_model = models.Model(inputs=[input1, input2], outputs=output_linear)

    return BSS_model









if __name__=="__main__":

    train_model = EncoderNetBigMel()
    print(train_model.summary())
    from keras.utils import plot_model
    plot_model(train_model, to_file='Model.png')
