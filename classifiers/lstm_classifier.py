from conf import conf
import os
from keras.layers import Dense, Activation, Dropout, Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
import librosa

class LSTMVoiceGenderClassifier(object):
    def __init__(self):
        self.hidden_unit = 512
        self.num_input_tokens = 512
        self.nb_classes = 2
        self.model_weight_file = "voice_gender_classifier"
        self.model = self.loadModel()

    def create_model(self):
        print("Building LSTM model....")
        model = Sequential()
        model.add(
            LSTM(units=self.hidden_unit, input_shape=(None, self.num_input_tokens), return_sequences=False, dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def load_model(self):
        model_dir = conf['MODEL_DIR']
        weight_file_path = os.path.join(model_dir, self.model_weight_file)
        self.model = self.create_model()
        if os.path.isfile(weight_file_path):
            print("Loading model weight...!!")
            self.model.load_weights(weight_file_path)

    def get_train_data(self):
        male_voice_arr = self.get_male_filelist()
        for file_path in male_voice_arr:
            x, sr = librosa.load(file_path)


    def fit(self):
        x_samples, y_sample = self.get_train_data()
