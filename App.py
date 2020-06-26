'''
    Created by wiktorkubis on Jun, 2020

'''
import os
import cv2
import time
import pygubu
import pygubu.builder.widgets.pathchooserinput
import numpy as np
import shutil
import threading
from PIL import Image, ImageTk
from pygubu.builder.widgets.pathchooserinput import PathChooserInput
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ImageGenerator import MobiTransfer, ProgressBarCallback

IMAGE_SIZE = (224, 224, 3)
COLLECTING_TIME = 4  # 25s of collecting data
EPOCHS = 1
BATCH_SIZE = 8


# TODO Add 'minus' button to delete some classes

class App(pygubu.TkApplication):

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 500)
        self.cap.set(4, 500)
        self.builder = pygubu.Builder()
        self.builder.add_from_file('myApp.xml')
        self.mainwindow = self.builder.get_object('Main')

        self.last_click = None
        self.trained = False
        self.training = False

        self._createRequiredDirs()

        super().__init__(self.mainwindow)

    def _create_ui(self):
        self.mainmenu = self.builder.get_object('menu')
        self.set_menu(self.mainmenu)

        self.imgLabel = self.builder.get_object('imgLabel')
        self.treeView = self.builder.get_object('treeView')
        self.labelTextField = self.builder.get_object('classEntry')
        self.counter = self.builder.get_object('Counter')
        self.progressBar = self.builder.get_object('progressBar')
        self.pathChooser = PathChooserInput()

        self.addButton = self.builder.get_object('ClassButton')
        self.trainButton = self.builder.get_object('TrainButton')

        callbacks = {
            'train': self.trainThread,
            'addClass': self.addClassThread,
            'menuLabels': self.menuLabelsThread,
        }

        self.builder.connect_callbacks(callbacks)
        self.mainwindow.protocol("WM_DELETE_WINDOW", self.on_close_window)

    def run(self):
        self.update()
        self.mainwindow.mainloop()

    def _createRequiredDirs(self):
        required_paths = ['Data', 'Data/Train', 'Data/Validation']

        for path in required_paths:
            if not os.path.isdir(path):
                os.mkdir(path)

    def trainThread(self):
        return threading.Thread(target=self.train).start()

    def addClassThread(self):
        return threading.Thread(target=self.addClass).start()

    def menuLabelsThread(self, menuItem):
        return threading.Thread(target=self.menuLabels, args=(menuItem,)).start()

    def addData(self, data, label_name):
        if self.iterator % 10 == 0:
            cv2.imwrite(f'Data/Validation/{label_name}/frame{self.iterator}.jpg', data)
        else:
            cv2.imwrite(f'Data/Train/{label_name}/frame{self.iterator}.jpg', data)
        self.iterator += 1

    def train(self):
        self.trainButton['state'] = 'disabled'
        self.training = True

        flow_params = {'target_size': IMAGE_SIZE[:2],
                       'color_mode': 'rgb',
                       'class_mode': 'categorical',
                       'batch_size': BATCH_SIZE,
                       'shuffle': True}

        imgGen = ImageDataGenerator(rescale=1. / 255,
                                    zoom_range=0.2,
                                    vertical_flip=True,
                                    shear_range=0.2
                                    )
        self.testGen = ImageDataGenerator(rescale=1. / 255)

        train_gen = imgGen.flow_from_directory(directory='Data/Train',
                                               **flow_params)

        val_gen = imgGen.flow_from_directory(directory='Data/Validation',
                                             **flow_params)

        self.labels = train_gen.class_indices
        self.progressBar['maximum'] = train_gen.samples

        self.model = MobiTransfer(IMAGE_SIZE, train_gen.num_classes)

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen,
                       callbacks=[ProgressBarCallback(self.progressBar, BATCH_SIZE)])
        self.training = False
        self.trained = True
        self.trainButton['state'] = 'normal'

    def updatePredictions(self, img):
        img = self.testGen.flow(img[np.newaxis, ...])
        predicted = self.model.predict(img)

        for child in self.treeView.get_children():
            item = self.treeView.item(child)
            index = self.labels[item['text']]
            proba = f'{round(predicted[0][index]*100, 2)}%'
            self.treeView.set(child, column=0, value=proba)
            if index == np.argmax(predicted):
                self.treeView.focus(child)
                self.treeView.selection_set(child)

    def menuLabels(self, menuItem):
        if menuItem == 'import':
            self.pathChooser.on_folder_btn_pressed()
        elif menuItem == 'export':
            self.pathChooser.on_folder_btn_pressed()

    def update(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        self.imgLabel.imgtk = imgtk
        self.imgLabel.configure(image=imgtk)

        img = np.array(img.resize(IMAGE_SIZE[:2]))

        if self.last_click:
            label_name = self.labelTextField.get()
            if time.time() - self.last_click < COLLECTING_TIME:
                self.counter['text'] = f'Adding {label_name}: {abs(int(time.time() - self.last_click) - COLLECTING_TIME)}'
                self.addData(cv2image, label_name)
            else:
                self.last_click = None
                self.counter['text'] = ''

        elif self.trained:
            self.updatePredictions(img)
            self.counter['text'] = ''
        elif self.training:  # TODO: add animation of dots
            self.counter['text'] = 'Training'

        else:
            self.addButton['state'] = 'normal'

        self.mainwindow.after(10, self.update)

    def addClass(self):
        self.iterator = 0  # TODO Add import option to import folder with labels
        self.last_click = time.time()
        self.addButton['state'] = 'disabled'
        label_name = self.labelTextField.get()
        try:
            os.mkdir(f'Data/Train/{label_name}')
            os.mkdir(f'Data/Validation/{label_name}')
        except FileExistsError:
            pass

        self.treeView.insert('', 1, text=label_name)

    def on_close_window(self):
        if input("Do you want delete files  y/n?") == 'y':
            for dir_ in os.listdir('Data/Train'):
                try:
                    shutil.rmtree(f'Data/Train/{dir_}')
                    shutil.rmtree(f'Data/Validation/{dir_}')
                except NotADirectoryError:
                    os.remove(f'Data/Train/{dir_}')
                    os.remove(f'Data/Validation/{dir_}')

        self.mainwindow.destroy()


if __name__ == '__main__':
    app = App()
    app.run()
