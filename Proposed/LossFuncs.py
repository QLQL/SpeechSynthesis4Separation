from keras import backend as K

def nonlinearTFweight(x):
    """
    x: NumSample x 100 x W, each item in the range of [0,1]
    """
    return K.pow(x, 2)
    # return K.sqrt(x)


def my_loss(y_true, y_pred):
    """
    y_true: NumSample x T x W x 3 # 1 with spectrum 2 mask labels
    y_pred: NumSample x T x W x 1 # the estimated spectrum
    """
    spectrum_gt = y_true[:, :, :, 0]
    spectrum_et = y_pred[:, :, :, 0]

    diff2_spectrum = K.square(spectrum_et - spectrum_gt)

    weight1 = nonlinearTFweight(spectrum_gt)
    weight2 = nonlinearTFweight(spectrum_et)

    weight_spectrum = weight1 + (1 - weight1) * weight2

    weight_sum_spectrum = K.sum(K.sum(weight_spectrum, axis=-1), axis=-1) + K.epsilon()
    weight_diff2_spectrum_sum = K.sum(K.sum(diff2_spectrum * weight_spectrum, axis=-1), axis=-1)

    output = weight_diff2_spectrum_sum / weight_sum_spectrum

    return output


def my_loss_mask(y_true, y_pred):
    """
    y_true: NumSample x Frame x W x 3       1-gt spectrum       2-gt mask       3-mixture spectrum
    y_pred: NumSample x Frame x W x 2       1-et spectrum       2-es mask
    """
    # print(y_true.shape)
    # print(y_pred.shape)


    spectrum_gt = y_true[:, :, :, 0]
    mask_gt = y_true[:, :, :, 1]
    mix_gt = y_true[:, :, :, 2]

    spectrum_et = y_pred[:, :, :, 0]
    mask_et = y_pred[:, :, :, 1]

    diff2_spectrum = K.square(spectrum_et - spectrum_gt)
    diff2_mask = K.square(mask_et - mask_gt)

    weight_mask = nonlinearTFweight(mix_gt)

    # weight1 = nonlinearTFweight(spectrum_gt)
    weight2 = nonlinearTFweight(spectrum_et)

    weight_spectrum = (weight_mask + (1-weight_mask)*weight2)*(1-mask_et)

    weight_sum_spectrum = K.sum(K.sum(weight_spectrum, axis=-1), axis=-1) + K.epsilon()
    weight_sum_mask = K.sum(K.sum(weight_mask, axis=-1), axis=-1) + K.epsilon()


    weight_diff2_spectrum_sum = K.sum(K.sum(diff2_spectrum * weight_spectrum, axis=-1), axis=-1)
    weight_diff2_mask_sum = K.sum(K.sum(diff2_mask * weight_mask, axis=-1), axis=-1)

    output = weight_diff2_spectrum_sum/weight_sum_spectrum + weight_diff2_mask_sum/weight_sum_mask

    # output = K.mean(K.mean(diff2_mask, axis=-1), axis=-1)

    return output



def my_loss_DC(Y, V):
    # Y size [BATCH_SIZE, FrameN, FrequencyN, 2] for two signal situations
    # V size [BATCH_SIZE, FrameN, FrequencyN, EMBEDDINGS_DIMENSION]
    def norm(tensor):
        square_tensor = K.square(tensor)
        frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
        return frobenius_norm2

    def dot(x, y):
        return K.batch_dot(x, y, axes=(2, 1))

    def T(x):
        return K.permute_dimensions(x, [0, 2, 1])

    # BATCH_SIZE = Y._keras_shape[0]
    MAX_MIX = 2 # either target or inteference
    EMBEDDINGS_DIMENSION = 40 # change this later
    Nframe = 64
    FrequencyN = 512
    temp = K.reshape(V, [-1, Nframe*FrequencyN, EMBEDDINGS_DIMENSION])
    V = K.l2_normalize(temp, axis=-1)

    Y = K.reshape(Y, [-1, Nframe * FrequencyN, MAX_MIX])

    silence_mask = K.sum(Y, axis=2, keepdims=True)
    V = silence_mask * V

    lossSum = norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))

    return lossSum/((Nframe*FrequencyN)**2)



# The only difference to my_loss_DC is the frequency number = 80 in the mel-scale
def my_loss_DC_mel(Y, V):
    # Y size [BATCH_SIZE, FrameN, FrequencyN, 2] for two signal situations
    # V size [BATCH_SIZE, FrameN, FrequencyN, EMBEDDINGS_DIMENSION]
    def norm(tensor):
        square_tensor = K.square(tensor)
        frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
        return frobenius_norm2

    def dot(x, y):
        return K.batch_dot(x, y, axes=(2, 1))

    def T(x):
        return K.permute_dimensions(x, [0, 2, 1])

    # BATCH_SIZE = Y._keras_shape[0]
    MAX_MIX = 2 # either target or inteference
    EMBEDDINGS_DIMENSION = 40 # change this later
    Nframe = 64
    FrequencyN = 80
    temp = K.reshape(V, [-1, Nframe*FrequencyN, EMBEDDINGS_DIMENSION])
    V = K.l2_normalize(temp, axis=-1)

    Y = K.reshape(Y, [-1, Nframe * FrequencyN, MAX_MIX])

    silence_mask = K.sum(Y, axis=2, keepdims=True)
    V = silence_mask * V

    lossSum = norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))

    return lossSum/((Nframe*FrequencyN)**2)




