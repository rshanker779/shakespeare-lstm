import os
import random
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import nltk
from nltk.corpus import shakespeare
import numpy as np


class GlobalParams:
    """Holder for general params"""
    model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model', 'shakespeare.h5')
    train_model = False


class Dimensions:
    """Holder for params of our data set"""
    p = 1
    chunks = 10
    num_classes = 1


def get_play_str(play):
    """Loads a Shakespeare play from nltk, name must match xml names of plays in nltk"""
    try:
        play = shakespeare.xml(play)
        full_str = ''.join(play.itertext())
        full_str = full_str.replace('\n', ' ')
        return full_str
    except LookupError:
        nltk.download('shakespeare')
        get_play_str(play)


def encode_chars(input_str):
    """One hot encode our character set. Could alternatively use scikit learn OneHotEncoder"""
    num_chars = sorted(list(set(input_str)))
    char_map = {j: i for i, j in enumerate(num_chars)}
    return char_map


def split_str_to_sequences(input_str, chunks, char_map):
    """Takes our long input string, splits into chunks we will feed to model and
    encodes character using our map."""
    n = len(input_str)
    p = len(char_map)
    train_x = np.zeros((n - chunks, chunks, p))
    train_y = np.zeros((n - chunks, p))
    for i in range(chunks, n):
        str_chunk = input_str[i - chunks:i]
        for index, j in enumerate(str_chunk):
            train_x[i - chunks, index, char_map[j]] = 1
        train_y[i - chunks, char_map[input_str[i]]] = 1
    return train_x, train_y


def get_lstm(train_x, train_y, chunks, p, num_classes, filepath=None):
    """Trains and saves our LSTM model"""
    model = Sequential()
    model.add(LSTM(120, input_shape=(chunks, p)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=20)
    if filepath is not None:
        model.save(filepath)
    return model


def generate_random_text(model, char_map, inital_str, output_size, chunk):
    """Take a string, takes the last chunk and repeatedly feeds it through the model"""
    output_str = inital_str[:]
    inverse_char_map = {j: i for i, j in char_map.items()}
    for _ in range(output_size):
        next_str = output_str[-chunk:]
        next_char_index = get_next_prediction(model, char_map, next_str, chunk)
        next_char = inverse_char_map[next_char_index]
        output_str += next_char
    return output_str


def get_next_prediction(model, char_map, inital_str, chunk):
    """Take our input string, get model's probabilistic predictions, and pick
    next character according to those probabilities"""
    initial_y = np.zeros((1, chunk, len(char_map)))
    for index, j in enumerate(inital_str):
        initial_y[:, index, char_map[j]] = 1
    char_array = model.predict(initial_y)
    cum_array = np.cumsum(char_array)
    ran_num = random.random()
    next_char_index = next(i[0] for i, j in np.ndenumerate(cum_array) if ran_num < j)
    return next_char_index


def get_simple_prediction(input: str, output_len: int = 1000):
    if len(input) < Dimensions.chunks:
        input += ' ' * (Dimensions.chunks - len(input))
    model = load_model(GlobalParams.model_save_path)
    char_map, _ = get_char_map()
    return generate_random_text(model, char_map, input[:Dimensions.chunks], output_len, Dimensions.chunks)


def main():
    char_map, play_str = get_char_map()
    Dimensions.num_classes = len(char_map)
    Dimensions.p = len(char_map)
    train_x, train_y = split_str_to_sequences(play_str, Dimensions.chunks, char_map)
    if not GlobalParams.train_model:
        try:
            print('Loading model')
            model = load_model(GlobalParams.model_save_path)
        except OSError:
            print('Model not found, retraining')
            model = get_lstm(train_x, train_y, Dimensions.chunks, Dimensions.p, Dimensions.num_classes,
                             GlobalParams.model_save_path)
    else:
        model = get_lstm(train_x, train_y, Dimensions.chunks, Dimensions.p, Dimensions.num_classes,
                         GlobalParams.model_save_path)

    output_str = generate_random_text(model, char_map, 'Romeo and ', 1000, Dimensions.chunks)
    print(output_str)


def get_char_map():
    play_str = get_play_str('dream.xml')
    char_map = encode_chars(play_str)
    return char_map, play_str


if __name__ == '__main__':
    main()
