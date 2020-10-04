
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
import numpy as np 

data = open("sonnets.txt").read()
corpus = data.lower().split("\n")


tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
total_words = len(word_index) + 1

#create list of sequence using list of tokens
#figuring out 'maxlen' (max length) of input_sequences. Would come in handy latter for padding!
input_sequences = []
maxlen = None
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[: i+1]
        input_sequences.append(n_gram)
        if maxlen == None or len(token_list) > maxlen:
            maxlen = len(token_list)

#padding the sequence
padding = 'pre'
maxlen = maxlen
#converting list to np.array is valuable, as it is compatible with tensorflow and slicing is easier
sequence_padding = np.array(pad_sequences(input_sequences, padding=padding, maxlen = maxlen))

#getting labels and predictor
predictor, labels = sequence_padding[:, :-1], sequence_padding[:, -1]
#converting labels to one-hot vector
labels = ku.to_categorical(labels, num_classes=total_words)



#Model Architecture
def build_model(total_words=total_words, maxlen=maxlen):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=maxlen-1))
    model.add(Bidirectional(LSTM(150, return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#Plotting training loss and accuracy
def plot():
    import matplotlib.pyplot as plt
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()

def get_words(model, seed_text, next_word=25):
    while next_word > 0:
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], padding=padding, maxlen=maxlen-1)
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word=""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text = seed_text + " " + output_word
        next_word = next_word - 1
    return seed_text

if __name__ == "__main__":
    history = build_model().fit(predictor, labels, epochs=100, verbose=1)

