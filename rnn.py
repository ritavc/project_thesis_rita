from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import keras
from keras.optimizers import SGD

from matplotlib import pyplot
import sys

from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr

df = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items.csv')
df["event"] = df["event"].astype('category')
df["event_categorical"] = df["event"].cat.codes
df = df.drop(['event'], axis=1)
df.sort_values(by=['timestamp'], inplace=True)

y = df['value']
X = df.drop(['value', 'id', 'property'], axis=1)
X = StandardScaler().fit_transform(X)
#y = StandardScaler().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
# for r in m: print(' '.join('{0:.5f}'.format(x) for x in r))

#print(X_train.head(15))

#model = Sequential([


opt = SGD(lr=0.01, momentum=0.9)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy', optimizer=opt
)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, verbose=0)
# evaluate the model
train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
# plot loss during training
pyplot.title('Loss / Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()