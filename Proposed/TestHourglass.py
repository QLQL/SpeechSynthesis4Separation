import keras
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Input, Add, MaxPool2D, Activation, Concatenate
from keras.models import Model
from keras import models
# from utils.config import num_stages, num_joints

num_stages = 1
num_joints = 2

nFeatures = 256
nStack = 1
nModules = 1
nOutChannels = num_joints

def convBlock(x, numOut):

    SC = models.Sequential()
    SC.add(BatchNormalization())
    # SC.add(Activation(activation='relu'))
    # SC.add(Conv2D(filters=int(numOut / 2), kernel_size=(1, 1), padding='same'))
    # SC.add(BatchNormalization())
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
    else:
        low2 = low1
        for i in range(nModules): low2 = Residual(low2, f)

    low3 = low2
    for i in range(nModules):
        low3 = Residual(low3, f)

    up2 = UpSampling2D(size=(2, 2))(low3)

    # Bring two branches together
    return Add()([up1, up2])

def lin(inp, numOut):
    l = Conv2D(numOut, (1,1), padding='same')(inp)
    l = BatchNormalization()(l)
    l = Activation(activation='relu')(l)
    return l

def hg_train():
    # input = Input(shape = (None, None, 3))
    input = Input(shape=(64, 512, 3))

    cnv1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(input)
    cnv1 = BatchNormalization()(cnv1)
    cnv1 = Activation(activation='relu')(cnv1)

    r1 = Residual(cnv1, 128)
    pool = MaxPool2D((2, 2), strides=2)(r1)
    r4 = Residual(pool, 128)
    r5 = Residual(r4, nFeatures)

    out = []
    inter = r5

    for i in range(nStack):
        hg = hourglass(3, nFeatures, inter)

        # Residual layers at output resolution
        ll = hg
        for j in range(nModules):
            ll = Residual(ll, nFeatures)

        ll = lin(ll, nFeatures)
        tmpOut = Conv2D(nOutChannels, (1,1), padding='same', name='stack_%d'%(i))(ll)
        out.append(tmpOut)

        if i < nStack:
            ll_ = Conv2D(nFeatures, (1, 1), padding='same')(ll)
            tmpOut_ = Conv2D(nFeatures, (1, 1), padding='same')(tmpOut)
            inter = Add()([inter, ll_, tmpOut_])

    model = Model(inputs=input, outputs=out)
    return model

if __name__ == '__main__':
    model = hg_train()
    print(model.summary())

    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='hg-f.png')
