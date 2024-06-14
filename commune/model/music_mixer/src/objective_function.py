from scipy.fftpack import fft
from scipy.fftpack import ifft
import numpy as np
import random
########## Constants ##########

# Frequency for equal-tempered scales
A0_FREQUENCY = 27.50
A0_SHARP_FREQUENCY = 29.14
A0_FLAT_FREQUENCY = 25.96
B0_FREQUENCY = 30.87
B0_FLAT_FREQUENCY = A0_SHARP_FREQUENCY
C0_FREQUENCY = 16.35
C0_SHARP_FREQUENCY = 17.32
D0_FREQUENCY = 18.35
D0_SHARP_FREQUENCY = 19.45
D0_FLAT_FREQUENCY = C0_SHARP_FREQUENCY
E0_FREQUENCY = 20.60
E0_FLAT_FREQUENCY = D0_SHARP_FREQUENCY
F0_FREQUENCY = 21.83
F0_SHARP_FREQUENCY = 23.12
G0_FREQUENCY = 24.50
G0_SHARP_FREQUENCY = A0_FLAT_FREQUENCY
G0_FLAT_FREQUENCY = F0_SHARP_FREQUENCY
SAMPLING_RATE = 44100

# Major Notes
A_MAJOR = "A_MAJOR"
A_FLAT_MAJOR = "A_FLAT_MAJOR"
A_SHARP_MAJOR = "A_SHARP_MAJOR"
B_MAJOR = "B_MAJOR"
B_FLAT_MAJOR = "B_FLAT_MAJOR"
C_MAJOR = "C_MAJOR"
C_SHARP_MAJOR = "C_SHARP_MAJOR"
D_MAJOR = "D_MAJOR"
D_SHARP_MAJOR = "D_SHARP_MAJOR"
D_FLAT_MAJOR = "D_FLAT_MAJOR"
E_MAJOR = "E_MAJOR"
E_FLAT_MAJOR = "E_FLAT_MAJOR"
F_MAJOR = "F_MAJOR"
F_SHARP_MAJOR = "F_SHARP_MAJOR"
G_MAJOR = "G_MAJOR"
G_SHARP_MAJOR = "G_SHARP_MAJOR"
G_FLAT_MAJOR = "G_FLAT_MAJOR"
frequencies = [C0_FREQUENCY, D0_FLAT_FREQUENCY, D0_FREQUENCY, E0_FLAT_FREQUENCY, E0_FREQUENCY, F0_FREQUENCY, G0_FLAT_FREQUENCY, G0_FREQUENCY, A0_FLAT_FREQUENCY, A0_FREQUENCY, B0_FLAT_FREQUENCY, B0_FREQUENCY]
names = [C_MAJOR, D_FLAT_MAJOR, D_MAJOR, E_FLAT_MAJOR, E_MAJOR, F_MAJOR, G_FLAT_MAJOR, G_MAJOR, A_FLAT_MAJOR, A_MAJOR, B_FLAT_MAJOR, B_MAJOR]

CHORDS = [set([C_MAJOR, E_MAJOR, G_MAJOR]), set([D_FLAT_MAJOR, F_MAJOR, A_FLAT_MAJOR]), set([D_MAJOR, G_FLAT_MAJOR, A_MAJOR]), set([E_FLAT_MAJOR, G_MAJOR, B_FLAT_MAJOR]), set([E_MAJOR, A_FLAT_MAJOR, B_MAJOR]),
    set([F_MAJOR, A_MAJOR, C_MAJOR]), set([G_FLAT_MAJOR, B_FLAT_MAJOR, D_FLAT_MAJOR]), set([G_MAJOR, B_MAJOR, D_MAJOR]), set([A_FLAT_MAJOR, C_MAJOR, E_FLAT_MAJOR]), set([A_MAJOR, D_FLAT_MAJOR, E_MAJOR]),
    set([B_FLAT_MAJOR, D_MAJOR, F_MAJOR]), set([B_MAJOR, E_FLAT_MAJOR, G_FLAT_MAJOR])]

SCALES = [(A_MAJOR, B_MAJOR,C_MAJOR, D_MAJOR, E_MAJOR, F_MAJOR, G_MAJOR),
          (A_SHARP_MAJOR, B_MAJOR, C_SHARP_MAJOR, D_SHARP_MAJOR, F_MAJOR, F_SHARP_MAJOR, G_SHARP_MAJOR),
          (A_MAJOR, B_MAJOR, C_SHARP_MAJOR, D_MAJOR, E_MAJOR,  F_SHARP_MAJOR, G_MAJOR),
          (),
          (A_MAJOR, B_MAJOR, C_SHARP_MAJOR, E_MAJOR, F_SHARP_MAJOR, G_SHARP_MAJOR),
          (A_MAJOR, B_FLAT_MAJOR, C_MAJOR, D_MAJOR, E_MAJOR, F_MAJOR, G_MAJOR),
          (A_SHARP_MAJOR, B_MAJOR, C_SHARP_MAJOR, D_SHARP_MAJOR, F_MAJOR, F_SHARP_MAJOR, G_SHARP_MAJOR),
          (A_MAJOR, B_MAJOR, C_MAJOR, D_MAJOR, E_MAJOR, F_SHARP_MAJOR, G_MAJOR),
          (),
          (A_MAJOR, B_MAJOR, C_SHARP_MAJOR, D_MAJOR, E_MAJOR, F_SHARP_MAJOR, G_SHARP_MAJOR),
          (B_MAJOR, C_SHARP_MAJOR, D_SHARP_MAJOR, E_MAJOR, F_SHARP_MAJOR, G_SHARP_MAJOR, A_SHARP_MAJOR, B_MAJOR)
          ]
SCALE_NAMES = [C_MAJOR, C_SHARP_MAJOR, D_MAJOR, D_SHARP_MAJOR, E_MAJOR, F_MAJOR, F_SHARP_MAJOR,
               G_MAJOR, G_SHARP_MAJOR, A_MAJOR, A_SHARP_MAJOR, B_MAJOR]

circle_of_fifths = {C_MAJOR:[G_MAJOR, F_MAJOR], G_MAJOR:[C_MAJOR, D_MAJOR],
                    D_MAJOR:[G_MAJOR, A_MAJOR], A_MAJOR:[D_MAJOR, E_MAJOR],
                    E_MAJOR:[A_MAJOR, B_MAJOR], B_MAJOR:[E_MAJOR, F_SHARP_MAJOR, G_FLAT_MAJOR],
                    F_SHARP_MAJOR:[B_MAJOR, C_SHARP_MAJOR, D_FLAT_MAJOR],
                    G_FLAT_MAJOR:[B_MAJOR, C_SHARP_MAJOR, D_FLAT_MAJOR],
                    C_SHARP_MAJOR:[G_FLAT_MAJOR, F_SHARP_MAJOR, A_FLAT_MAJOR],
                    D_FLAT_MAJOR:[G_FLAT_MAJOR, F_SHARP_MAJOR, A_FLAT_MAJOR],
                    A_FLAT_MAJOR:[D_FLAT_MAJOR, C_SHARP_MAJOR,E_FLAT_MAJOR],
                    E_FLAT_MAJOR:[A_FLAT_MAJOR, B_FLAT_MAJOR], B_FLAT_MAJOR:[E_FLAT_MAJOR, F_MAJOR],
                    F_MAJOR:[B_FLAT_MAJOR, C_MAJOR]}

equivalent_keys = {A_SHARP_MAJOR:B_FLAT_MAJOR, B_FLAT_MAJOR:A_SHARP_MAJOR,
                   G_SHARP_MAJOR:A_FLAT_MAJOR, A_FLAT_MAJOR:G_SHARP_MAJOR,
                   F_SHARP_MAJOR:G_FLAT_MAJOR, G_FLAT_MAJOR: F_SHARP_MAJOR,
                   C_SHARP_MAJOR:D_FLAT_MAJOR, D_FLAT_MAJOR:C_SHARP_MAJOR,
                   D_SHARP_MAJOR:E_FLAT_MAJOR, E_FLAT_MAJOR:D_SHARP_MAJOR,
                    }

###############################

def score_on_chord(chord, preferred_chord):
    if chord == preferred_chord:
        return 1
    else:
        return 0.1

def score_on_key_signature(key_sig, preferred_key_sig):
    # if key_sig == preferred_key_sig or (key_sig in equivalent_keys and equivalent_keys[preferred_key_sig]):
    if key_sig == preferred_key_sig:
        return 1
    elif key_sig in circle_of_fifths[preferred_key_sig]:
        return 0.5
    else:
        return 0.1

def get_scale(notes):
    for i,s in enumerate(SCALES):
        if sorted(s) == sorted(notes):
            return SCALE_NAMES[i]

def get_chord(notes):
    set_notes = set(notes)
    for i in notes:
        chord = CHORDS[names.index(i)]
        if chord.issubset(set_notes):
            return i
    return ""

def compute_intensities(starting_frequency, fft_view):
    intensities = np.array([])
    freq = starting_frequency
    for _ in range(9):
        intensities = np.append(intensities,np.real(fft_view[round(freq * len(fft_view) / SAMPLING_RATE)]))
        freq*=2
    return intensities



def fitness(f, preferred_chord, preferred_scale):
    '''
    Determine fitness of some sound sample f

    parameters:
        f: sound sample in spatial domain

    return:
        fitness value of the sound sample according to various music theory
        metrics
    '''
    score = notes_intensity(f, preferred_chord, preferred_scale)
    return score


def find_nearest(array, value):
    array = np.asarray(array)

    idx = (np.abs(array - value)).argmin()
    return idx

def convert(f):
    notes = []
    index = 0
    while(len(notes)<8):
        temp = f[index]
        while(temp>32):
            temp=temp/2

        name = names[find_nearest(frequencies,temp)]
        if name not in notes:
            notes.append(name)

        index+=1
        if(index > 128):
            break
    return notes


def notes_intensity(f, preferred_chord, preferred_scale):
    '''
    Determine the prescense of notes A, B, C, D, E, F, G across various
    octaves
    '''

    fft_view = np.abs((fft(f)))
    x = len(fft_view)
    if(len(fft_view) % 2 == 1):
        chop = int(len(fft_view) / 2) + 1
    else:
        chop = int(len(fft_view) / 2)
    fft_view=fft_view[1:chop]

    y = fft_view.argsort()[::-1]* 44100/x

    notes = convert(y)


    chord = get_chord(notes)
    chord_score = score_on_chord(chord, preferred_chord)
    key_sig_score = score_on_key_signature(notes[0], preferred_scale)
    return chord_score + key_sig_score