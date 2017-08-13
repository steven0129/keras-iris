import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import pandas as pd
import numpy as np
from urllib.request import urlopen
import lib.function as lib
from keras.wrappers.scikit_learn import KerasClassifier

response = urlopen(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
data = response.read()

if(not os.path.isfile('iris.csv')):
    file = open('iris.csv', 'wb')
    file.write(data)
    file.close()

# load dataset
myDataframe = pd.read_csv('iris.csv', header=None)
myDataframe = myDataframe.sample(frac=1).reset_index(drop=True)  # 打亂data
Xtrain, Ytrain, Xtest, Ytest = lib.preprocess(myDataframe)

# build model
myModel = lib.model2()
history = myModel.fit(Xtrain, Ytrain, epochs=400,
                      validation_split=0.2, verbose=0)

# evaluate model
myModel.evaluate(Xtest, Ytest)

# predict by model
result = lib.predict([5.1, 3.5, 1.4, 0.2], myModel)
print(result)

# save model
myModel.save('myModel.h5')
