'''
    Created by wiktorkubis on Jun, 2020

'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import Callback

IMAGE_SIZE = (224, 224, 3)

flow_params = {'target_size': (224, 224),
               'color_mode': 'rgb',
               'class_mode': 'categorical',
               'batch_size': 32,
               'shuffle': True}


def plot_history(history):
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    acc = history.history['accuracy']

    fig, ax = plt.subplots(1, 2, figsize=(25, 7))

    ax[0].plot(val_loss, label='val loss')
    ax[0].plot(loss, label='train loss')
    ax[0].set_ylim(0, 1)
    ax[0].legend()

    ax[1].plot(val_acc, label='val acc')
    ax[1].plot(acc, label='train acc')
    ax[1].legend()
    plt.show()



class ProgressBarCallback(Callback):
    def __init__(self, progressbar, step):
        self.progressbar = progressbar
        self.step = step


    def on_batch_end(self, batch, logs=None):

        self.progressbar.step(self.step)


class MobiTransfer(Model):
    def __init__(self, input_shape, outputs):
        super(MobiTransfer, self).__init__()
        self.outputs = outputs
        self.mobile = MobileNet(input_shape=input_shape, include_top=False, dropout=0.1)
        self.mobile.trainable = False

        self.flat = Flatten()
        self.Dense1 = Dense(outputs, activation ='softmax')

    def __call__(self, inputs, training = False):
        x = self.mobile(inputs)
        x = self.flat(x)
        return self.Dense1(x)


if __name__ == '__main__':
    imgGen = ImageDataGenerator(rescale=1 / 255,
                                horizontal_flip=True,
                                vertical_flip=True)

    train_gen = imgGen.flow_from_directory(directory='Data/Train', **flow_params)

    # val_gen = imgGen.flow_from_directory(directory='Data/Validation', **flow_params)
    #
    # mobilenet = MobileNet(input_shape=IMAGE_SIZE, include_top=False)
    # mobilenet.trainable = False
    # #
    # # inputs = Input(shape=IMAGE_SIZE)
    # # x = mobilenet(inputs)
    # # x = Flatten()(x)
    # # outputs = Dense(train_gen.num_classes, activation='softmax')(x)
    # # model = Model(inputs, outputs)
    # #
    # # model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # #
    # # history = model.fit(train_gen, epochs=10, validation_data=val_gen)
    #
    # # myModel = MobiTransfer(train_gen.num_classes)
    #
    # myModel = Sequential([
    #     mobilenet,
    #     Flatten(),
    #     Dense(train_gen.num_classes, activation='softmax')
    # ])

    # myModel.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # history = myModel.fit(train_gen, epochs=5, validation_data=val_gen)
    # myModel.save('myModel.h5')

    myModel = load_model('myModel.h5')

    test_img = next(train_gen)[0][0]

    predictions = myModel.predict()
    print(predictions)

    # plot_history(history)
