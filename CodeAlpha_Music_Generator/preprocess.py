import glob
import pickle
import music21
from music21 import converter, instrument, note, chord

notes = []

print("Reading MIDI files...")

for file in glob.glob("dataset/*.mid"):

    midi = converter.parse(file)

    parts = instrument.partitionByInstrument(midi)

    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:

        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print("Total notes extracted:", len(notes))

with open('notes', 'wb') as filepath:
    pickle.dump(notes, filepath)

print("Preprocessing completed.")