import os
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from classifiers.lstm_data_generator import DataGenerator
from keras.layers.recurrent import LSTM
from keras.models import load_model

class LSTMVoiceGenderClassifier(object):
    def __init__(self):
        self.hidden_unit = 512
        self.num_input_tokens = 512
        self.nb_classes = 2
        self.batch_size = 32
        self.input_dim = (self.batch_size, 40, )
        self.num_epochs = 10
        self.data_train_dir = "data/train"
        self.model_weight_file = "models/voice_gender_classifier.weight"

    def create_model(self):
        print("Building LSTM model....")
        model = Sequential()
        model.add(
            LSTM(units=self.hidden_unit, input_shape=self.input_dim, return_sequences=False, dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))


        return model

    def load_model(self):
        if os.path.isfile(self.model_weight_file):
            print("Loading model weight...!!")
            model = load_model(self.model_weight_file)
        else:
            model = self.create_model()
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def get_training_desc(self):
        file_path_list = []
        labels = {}
        for dir_name in os.listdir(self.data_train_dir):
            dir_path = os.path.join(self.data_train_dir, dir_name)
            label = 1
            if "female" in dir_name:
                label = 0
            for file in os.listdir(dir_path):
                full_file_path = os.path.join(dir_path, file)
                file_path_list.append(full_file_path)
                labels[full_file_path] = label

        x_train, x_test = train_test_split(file_path_list, test_size=0.2, random_state=2)
        return {'train': x_train, 'validation': x_test}, labels

    def fit(self):
        partition, labels = self.get_training_desc()
        params = {'dim': self.input_dim,
                  'batch_size': self.batch_size,
                  'n_classes': self.nb_classes,
                  'shuffle': True}
        training_generator = DataGenerator(partition['train'], labels, **params)
        validation_generator = DataGenerator(partition['validation'], labels, **params)
        self.model = self.load_model()
        if not self.model:
            print("Cannot load model")
            return
        self.model.fit_generator(generator=training_generator, epochs=self.num_epochs, verbose=1,
                                           validation_data=validation_generator, callbacks=[])
        self.model.save(self.model_weight_file)
