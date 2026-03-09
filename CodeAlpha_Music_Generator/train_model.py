import numpy as np
import pickle
import tensorflow as tf

from tensorflow import Sequential
from tensorflow import Dense, Dropout, LSTM
from tensorflow import to_categorical

# sequence length
sequence_length = 100

print("Loading notes...")

# load notes
with open('notes', 'rb') as filepath:
    notes = pickle.load(filepath)

print("Total notes:", len(notes))

# unique notes
pitchnames = sorted(set(notes))
print("Unique notes:", len(pitchnames))

# mapping notes to numbers
note_to_int = {note: number for number, note in enumerate(pitchnames)}

network_input = []
network_output = []

# create training sequences
for i in range(0, len(notes) - sequence_length):

    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]

    network_input.append([note_to_int[n] for n in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

print("Total patterns:", n_patterns)

# reshape input for LSTM
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

# normalize
network_input = network_input / float(len(pitchnames))

# categorical output
network_output = to_categorical(network_output)

# build model
model = Sequential()

model.add(LSTM(
    256,
    input_shape=(network_input.shape[1], network_input.shape[2]),
    return_sequences=True
))

model.add(Dropout(0.3))

model.add(LSTM(256))

model.add(Dropout(0.3))

model.add(Dense(len(pitchnames), activation='softmax'))

# compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam'
)

print("Training model...")

# train model
model.fit(
    network_input,
    network_output,
    epochs=20,
    batch_size=64
)

# save model
model.save("music_model.h5")

print("Model saved successfully.")