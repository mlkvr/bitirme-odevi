from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from pandas import read_csv
# split a multivariate sequence into samples

def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

def split_sequences(sequences, n_steps):
    X, y = list(), list()

    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

conf_data = read_csv("trdata.csv", index_col=False, usecols=["confirmed"])
recv_data = read_csv("trdata.csv", index_col=False, usecols=["recoveries"])
outp_data = read_csv("trdata.csv", index_col=False, usecols=["deaths"])

items1 = conf_data.to_numpy(dtype=int)
items2 = recv_data.to_numpy(dtype=int)
targets = outp_data.to_numpy(dtype=int)

items1 = items1.reshape((len(items1), 1))
items2 = items2.reshape((len(items2), 1))
targets = targets.reshape((len(targets), 1))

dataset = hstack((items1, items2, targets))
# choose a number of time steps
n_steps = 1
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# flatten input
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[95591], [14918]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)