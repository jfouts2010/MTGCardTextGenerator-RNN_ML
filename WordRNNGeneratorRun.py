import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

def charGenerator():
    with open("mtgCardOutputAbilitiesV3.txt", "r") as f:
        text = f.read()
    starters = []
    with open("mtgCardOutputAbilitiesV3.txt", "r") as f:
        for line in f:
            if '<' in line:
                lessthan = line.index('<')
            else:
                lessthan = 9999
            if ' ' in line:
                space = line.index(' ')
            else:
                space = 9999
            if space == line:
                continue
            if lessthan < space:
                start = line[:lessthan]
                if start != '':
                    starters.append(start)
            else:
                start = line[:space]
                if start != '':
                    starters.append(start)
            #find index of first < or ' '
            x = 5
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(text)
    max_id= len(tokenizer.word_index)
    model = keras.models.load_model("testCHARMTGCardCreator.h5")
    def preprocess(text):
        X = np.array(tokenizer.texts_to_sequences(text)) - 1
        return tf.one_hot(X, max_id)

    Cards = []
    for a in range(300):
        predictedText = "<STARTOFCARD>"#random.choice(starters)
        for i in range(300):
            X_new = preprocess([predictedText])
            y_proba = model.predict(X_new)[0, -1:, :]
            rescaled_logits = tf.math.log(y_proba)/ 1
            char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
            prediction = tokenizer.sequences_to_texts(char_id.numpy())[0]
            predictedText += prediction
            if "<endofcard>" in predictedText:
                Cards.append(predictedText)
                break

    with open('generatedCards.txt', 'w') as f:
        for item in Cards:
            f.write("%s\n" % item)
    x  = 5

def wordGenerator():
    with open("mtgCardOutputV3.txt", "r") as f:
        text = f.read()
    res = tf.keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;=?@[\\]^_`|~\t\n',
                                                            lower=True, split=' ')
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(res)
    max_id = len(tokenizer.word_index)
    model = keras.models.load_model("test_WORD_MTGCardCreator.h5")


    def preprocess(text):
        X = np.array(tokenizer.texts_to_sequences(text)) - 1
        return tf.one_hot(X, max_id)


    Cards = []
    for a in range(300):
        predictedText = "<STARTOFCARD>"#random.choice(starters)
        for i in range(300):
            X_new = preprocess([predictedText])
            y_proba = model.predict(X_new)[0, -1:, :]
            rescaled_logits = tf.math.log(y_proba) / 1
            char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
            prediction = tokenizer.sequences_to_texts(char_id.numpy())[0]
            predictedText += prediction + " "
            if "endofcard" in predictedText:
                Cards.append(predictedText)
                break

    with open('generatedCards.txt', 'w') as f:
        for item in Cards:
            f.write("%s\n" % item)
    x = 5
charGenerator()