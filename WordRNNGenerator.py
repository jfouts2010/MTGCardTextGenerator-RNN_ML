import tensorflow as tf
from tensorflow import keras
import numpy as np


def CharRNN():
    with open("mtgCardOutputAbilitiesV3.txt") as f:
        text = f.read()
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(text)

    [encoded] = np.array(tokenizer.texts_to_sequences([text])) - 1
    max_id= len(tokenizer.word_index)
    dataset_size = tokenizer.document_count
    train_size = dataset_size * 90//100
    dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
    n_steps = 100
    window_length = n_steps +1
    dataset = dataset.window(window_length, shift=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    epochs = 1
    batch_size = 32
    dataset = dataset.shuffle(10000).batch(batch_size)
    dataset = dataset.map(lambda windows: (windows[:,:-1], windows[:,1:]))
    dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
    dataset = dataset.prefetch(1)
    for item in dataset:
        x = 5
    model = keras.models.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=[None, max_id],dropout=0.2, recurrent_dropout=0.2),
        keras.layers.LSTM(128, return_sequences=True,dropout=0.2, recurrent_dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))
                                     ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    history = model.fit(dataset, epochs=epochs)
    model.save("testCHARMTGCardCreator.h5")
    def preprocess(text):
        X = np.array(tokenizer.texts_to_sequences(text)) - 1
        return tf.one_hot(X, max_id)

    X_new = preprocess(["How are yo"])
    Y_pred = model.predict_classes(X_new)
    prediction = tokenizer.sequences_to_texts(Y_pred + 1)[0][-1]
    x  = 5


def WordRNN(): #ISSUE WITH EMBEDDING AND TOKEN WORDS LIKE ENDOFCARD, ECT
    with open("mtgCardOutputV3.txt") as f:
        text = f.read()

    res = tf.keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;=?@[\\]^_`|~\t\n', lower=True, split=' ')
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(res)

    [encoded] = np.array(tokenizer.texts_to_sequences([res])) - 1
    max_id = len(tokenizer.word_index)
    dataset_size = tokenizer.document_count
    train_size = dataset_size * 90 // 100
    dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
    n_steps = 30
    window_length = n_steps + 1
    dataset = dataset.window(window_length, shift=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    epochs = 1
    batch_size = 32
    dataset = dataset.shuffle(10000).batch(batch_size)
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
    dataset = dataset.prefetch(1)


    model = keras.models.Sequential(
        [keras.layers.LSTM(128, return_sequences=True, input_shape=[None, max_id], dropout=0.2, recurrent_dropout=0.2),
         keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
         keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))
         ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    history = model.fit(dataset, epochs=epochs)
    model.save("test_WORD_MTGCardCreator.h5")



CharRNN()