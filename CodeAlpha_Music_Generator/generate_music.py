import numpy as np
import pickle
import tensorflow
from tensorflow import Dense, Dropout, LSTM
from tensorflow import Sequential
from tensorflow import to_categorical

sequence_length = 100

with open('notes', 'rb') as filepath:
    notes = pickle.load(filepath)

pitchnames = sorted(set(notes))

note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length, 1):

    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]

    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(len(pitchnames))

network_output = to_categorical(network_output)

model = Sequential()

model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(256))
model.add(Dropout(0.3))

model.add(Dense(len(pitchnames), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print("Training model...")

model.fit(network_input, network_output, epochs=20, batch_size=64)

model.save("music_model.h5")

print("Model training completed.")