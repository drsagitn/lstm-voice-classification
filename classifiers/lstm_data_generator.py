import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
import librosa

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_batch):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, ), dtype=int)
        for i, filename in enumerate(list_IDs_batch):
            try:
                x, sample_rate = librosa.load(filename, res_type='kaiser_fast')
                X[i, ] = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0).reshape(40,1)
                y[i] = self.labels[filename]
            except Exception as e:
                print("Exception while reading", filename, " skipping it")
                print(e.message)
                continue
        encoder = LabelEncoder()
        y2 = encoder.fit_transform(y)
        Y = keras.utils.to_categorical(y2, num_classes=self.n_classes)
        return X, Y
