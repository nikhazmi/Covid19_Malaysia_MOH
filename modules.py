from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

def lstm_model_creation(output = 1, dropout=0.3, num_neurons= 64):
    """This function creates LSTM models with embedding layer, 2 LSTM layers, with dropout and _summary_

    Args:
        num_words (int): number of vocabulary
        nb_classes (int): number of class
        embedding_layer (int, optional): the number of output embedding llayer. Defaults to 64.
        dropout (float, optional): the rate of dropout. Defaults to 0.3.
        num_neurons (int, optional): number of rbain cells. Defaults to 64.

    Returns:
        model: returns the model created using sequential API.
    """
    model = Sequential()
    model.add(Input(shape = X_train.shape[1:]))
    model.add(LSTM(num_neurons, return_sequences= True))
    model.add(Dropout(dropout))
    model.add(LSTM(num_neurons))
    model.add(Dropout(dropout))
    model.add(Dense(output, activation = 'softmax'))
    model.summary()

    model.compile(optimizer = 'adam', loss = 'mse', metrics=['mse', 'mape'])
    plot_model(model)
    return model
