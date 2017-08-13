from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np


def preprocess(dataframe, train_proportion=0.8):
    dataset = dataframe.values
    X = dataset[:, 0:4].astype(float)
    Y = dataset[:, 4]

    encoder = LabelEncoder()
    encoder.fit(Y)
    classes = encoder.classes_
    Y2encode = encoder.transform(Y)
    Y = np_utils.to_categorical(Y2encode)

    sampleNum = len(dataset)
    Xtrain = X[:int(sampleNum * train_proportion)]
    Ytrain = Y[:int(sampleNum * train_proportion)]
    Xtest = X[int(sampleNum * train_proportion):]
    Ytest = Y[int(sampleNum * train_proportion):]
    return Xtrain, Ytrain, Xtest, Ytest


def predict(data, model):
    values = model.predict(np.array([data])).tolist()
    value = values[0]
    index = value.index(max(value))
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    return classes[index]


def pltLearningCurve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def model1():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def model2():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='softmax'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model
