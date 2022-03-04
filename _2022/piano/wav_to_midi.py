from manim_imports_ext import *

import argparse
import wave
import matplotlib.pyplot as plt

import mido
from collections import namedtuple
from tqdm import tqdm as ProgressDisplay

from scipy.signal import fftconvolve

from IPython.terminal.embed import InteractiveShellEmbed
embed = InteractiveShellEmbed()


SAMPLED_VELOCITY = 100
SAMPLED_VELOCITIES = list(range(25, 150, 25))
DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)
PIANO_SAMPLES_DIR = os.path.join(DATA_DIR, "piano_samples")
TEST_SPEECH = os.path.join(DATA_DIR, "IAmAPiano.wav")
# TEST_SPEECH = os.path.join(DATA_DIR, "SampleLetters.wav")
CLACK = "/Users/grant/Dropbox/3Blue1Brown/sounds/clack.wav"

Note = namedtuple(
    'Note',
    [
        'value',
        'velocity',
        'position',  # In seconds
        'duration',  # In seconds
    ]
)

major_scale = [0, 2, 4, 5, 7, 9, 11]
piano_midi_range = list(range(21, 109))


def square(vect):
    return np.dot(vect, vect)


def norm(vect):
    return np.linalg.norm(vect)

# Functions for creating MIDI files


def hz_to_midi(frequencies):
    freqs = np.atleast_1d(frequencies)
    return (12 * np.log2(freqs / 440) + 69).astype(int)


def midi_to_hz(midis):
    midis = np.atleast_1d(midis)
    return 440 * 2**((midis - 69) / 12)


def add_notes(track, notes, sec_per_tick):
    """
    Adapted from https://github.com/aniawsz/rtmonoaudio2midi
    """
    curr_tick = 0
    for index, note in enumerate(notes):
        pos_in_ticks = int(note.position / sec_per_tick)
        dur_in_ticks = int(note.duration / sec_per_tick)

        if index < len(notes) - 1:
            next_pos_in_ticks = int(notes[index + 1].position / sec_per_tick)
            dur_in_ticks = min(dur_in_ticks, next_pos_in_ticks - pos_in_ticks)

        track.append(
            mido.Message(
                'note_on',
                note=int(note.value),
                velocity=int(note.velocity),
                time=pos_in_ticks - curr_tick
            )
        )
        curr_tick = pos_in_ticks
        track.append(
            mido.Message(
                'note_off',
                note=int(note.value),
                # velocity=int(note.velocity),
                time=dur_in_ticks,
            )
        )
        curr_tick = pos_in_ticks + dur_in_ticks


def create_midi_file_with_notes(filename, notes, bpm=240):
    """
    From https://github.com/aniawsz/rtmonoaudio2midi
    """
    with mido.MidiFile() as midifile:
        # Tempo is microseconds per beat
        tempo = int((60.0 / bpm) * 1000000)
        sec_per_tick = tempo / 1000000.0 / midifile.ticks_per_beat

        # Create one track for each piano key
        tracks = []
        for key in piano_midi_range:
            track = mido.midifiles.MidiTrack()
            matching_notes = list(filter(lambda n: n.value == key, notes))
            matching_notes.sort(key=lambda n: n.position)
            if len(matching_notes) == 0:
                continue
            add_notes(track, matching_notes, sec_per_tick)
            tracks.append(track)

        master_track = mido.midifiles.MidiTrack()
        # master_track.append(mido.MetaMessage('instrument_name', name='Steinway Grand Piano', time=0))
        master_track.append(mido.MetaMessage('instrument_name', name='Learner\'s Piano', time=0))
        master_track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        master_track.extend(mido.merge_tracks(tracks))
        midifile.tracks.append(master_track)

        midifile.save(filename)


def midi_to_wav(mid_file):
    wav_file = mid_file.replace("mid", "wav")
    mp3_file = mid_file.replace("mid", "mp3")
    if os.path.exists(wav_file):
        os.remove(wav_file)
    os.system(" ".join([
        "timidity",
        mid_file,
        "-Ow -o -",
        "|",
        "ffmpeg",
        "-i - -acodec libmp3lame -ab 64k -hide_banner -loglevel error",
        mp3_file,
        "> /dev/null"
    ]))
    os.system(" ".join([
        "ffmpeg",
        "-hide_banner -loglevel error",
        "-i",
        mp3_file,
        wav_file,
    ]))
    os.remove(mp3_file)


def generate_pure_piano_key_files(velocities=[SAMPLED_VELOCITY], duration=1 / 96):
    folder = PIANO_SAMPLES_DIR
    if not os.path.exists(folder):
        os.makedirs(folder)

    for key in piano_midi_range:
        for vel in velocities:
            note = Note(key, vel, 0, duration)
            mid_file = os.path.join(folder, f"{key}_{vel}.mid")
            create_midi_file_with_notes(mid_file, [note])
            midi_to_wav(mid_file)
            os.remove(mid_file)


def generate_piano_sample_midi(delay=5, duration=1 / 96, velocity=100):
    notes = []
    for n, key in enumerate(piano_midi_range):
        notes.append(Note(
            key,
            velocity=velocity,
            position=n * delay,
            duration=duration,
        ))
    mid_file = os.path.join(DATA_DIR, "individual_key_samples.mid")
    create_midi_file_with_notes(mid_file, notes)
    midi_to_wav(mid_file)


def parse_piano_samples_with_delay(wav_file, target_folder, delay=5):
    pass


# Using fourier

def load_piano_key_signals(folder=PIANO_SAMPLES_DIR, duration=0.5, velocity=50):
    sample_rate = 48000
    key_signals = []
    for key in piano_midi_range:
        full_signal = wav_to_array(os.path.join(folder, f"{key}_{velocity}.wav"))
        vect = full_signal[:int(duration * sample_rate)]
        key_signals.append(vect)
    return np.array(key_signals, dtype=float)


def wav_to_midi(sound_file,
                duration=1 / 48,  # How to choose this?
                played_duration=1 / 48,
                sample_velocity=100,
                volume_ratio_threshold=0.75,
                max_volume=10000,
                steps_per_second=240,  # And how to choose this?
                n_repressed_lower_keys=32,  # Honestly, low keys are trash, just trashing up the whole sound
                max_notes_per_step=2,
                ):
    """
    Walk through a series of windows over the original signal, and for each one,
    find the top several key sounds which correlate most closely with that window.
    More specifically, do a convolution to let that piano key signal 'slide' along
    the window to find the best possible match.

    Room for improvement:
        - Duration shouldn't necessarily be fixed
    """
    sample_rate = 48000  # Should get this from the file itself
    step_size = int(sample_rate / steps_per_second)
    window_size = int(sample_rate * duration + step_size)

    notes = []
    key_signals = load_piano_key_signals(duration=duration, velocity=sample_velocity)

    # Read in audio file, and soften so as to never exceed piano samples
    signal = wav_to_array(sound_file).astype(float)
    signal *= max_volume / signal.max()  # Soften
    new_signal = np.zeros_like(signal)

    # To prevent keys from running over each other, keep track of the next available
    # spot when each note is allowed to be played.
    key_to_min_pos = {key: 0 for key in piano_midi_range}

    for pos in ProgressDisplay(range(0, len(signal), step_size), leave=False):
        window = signal[pos:pos + window_size]
        new_window = new_signal[pos:pos + window_size]
        window_diff = window - new_window

        sub_window = signal[pos:pos + step_size]
        new_sub_window = new_signal[pos:pos + step_size]

        # Find the best several keys to add in this window
        convs = np.array([
            fftconvolve(ks[::-1], window_diff, mode='valid')
            for ks in key_signals
        ])
        indices = np.argsort(convs.max(1))[::-1]
        indices = (
            *indices[indices > n_repressed_lower_keys],
            *reversed(range(n_repressed_lower_keys)),
        )

        for i in indices[:max_notes_per_step]:
            key = piano_midi_range[i]
            ks = key_signals[i]
            offset = convs[i].argmax()
            opt_pos = pos + offset

            # Check if we're allowed to use this key
            if key_to_min_pos[key] > opt_pos:
                continue

            win_norm = norm(sub_window)
            new_norm = norm(new_sub_window)

            # If this sub_window is as loud as that of the original signal, stop
            # adding new keys
            if new_norm > volume_ratio_threshold * win_norm:
                break

            # If the projection of segment onto ks is f * ks, this gives f
            segment = (window - new_window)[offset:offset + len(ks)]
            short_ks = ks[:len(segment)]
            factor = np.dot(segment, short_ks) / np.dot(short_ks, short_ks)
            factor = clip(factor, 0, 1)

            if factor > 0:
                # Add the key_signal to new_window, which will in turn be added to new_signal
                piece = new_window[offset:offset + len(ks)]
                piece += factor * ks[:len(piece)]
                # Mark this key as unavailable for the next len(ks) samples
                key_to_min_pos[key] = opt_pos + len(ks)
                # Add the note, which will ultimately be used to create the MIDI file
                notes.append(Note(
                    value=key,
                    velocity=clip(factor * sample_velocity, 0, 100),
                    position=opt_pos / sample_rate,
                    # Right now at least, it always hits with a short staccato
                    duration=played_duration,
                ))

    mid_file = sound_file.replace(".wav", "_as_piano.mid")
    create_midi_file_with_notes(mid_file, notes)
    midi_to_wav(mid_file)

    plt.plot(signal, linewidth=1.0)
    plt.plot(new_signal, linewidth=1.0)
    plt.show()

    return


def still_terrible_wav_to_midi_strat():
    # Old (bad) strat
    for n in range(20):
        indices = np.argsort(np.abs(np.array(piano_midi_range) - 69))  # Sort by closest to middle A
        for i in indices:
            ks = key_signals[i]
            v_norm = norm(ks)
            key = piano_midi_range[i]
            conv = fftconvolve(ks[::-1], signal)
            conv = conv[len(ks) - 1:]  # Only care about parts where piano sound overlaps fully
            opt_pos = np.argmax(conv)
            segment = signal[opt_pos:opt_pos + len(ks)]

            # If the projection of segment onto ks is f * ks, this gives f
            factor = conv[opt_pos] / (v_norm**2)
            # Cannot add more than max velocity key hit
            factor = min(factor, 127 / SAMPLED_VELOCITY)

            segment -= (factor * ks).astype(int)

            notes.append(Note(
                value=key,
                velocity=factor * SAMPLED_VELOCITY,
                position=opt_pos / sample_rate,
                duration=duration,
            ))


def previous_terrible_wav_to_midi():
    sample_rate = 48000  # Should get this from the file itself
    bucket_size = 1 / 60  # In seconds
    step = int(sample_rate * bucket_size)

    notes = []
    for n in ProgressDisplay(range(0, len(signal), step)):
        bucket = signal[n:n + step]
        times = np.linspace(0, len(bucket) / sample_rate, len(bucket))
        for key in piano_midi_range:
            freq = midi_to_hz(key)
            cos_wave = np.cos(TAU * freq * times)
            sin_wave = np.sin(TAU * freq * times)
            cos_wave /= norm(cos_wave)
            sin_wave /= norm(sin_wave)
            strength = get_norm([
                np.dot(cos_wave, bucket),
                np.dot(sin_wave, bucket),
            ])
            velocity = 2 * strength
            if velocity > 1:
                notes.append(Note(
                    value=key,
                    velocity=min(velocity, 127),
                    position=(n / step) * bucket_size,
                    duration=bucket_size,
                ))

    create_midi_file_with_notes(
        sound_file.replace(".wav", ".mid"),
        notes,
    )


# Functions for processing sound files


def wav_to_array(file_name):
    fp = wave.open(file_name)
    nchan = fp.getnchannels()
    N = fp.getnframes()
    dstr = fp.readframes(N * nchan)
    data = np.frombuffer(dstr, np.int16)
    data = np.reshape(data, (-1, nchan))
    data = data[:, 0].copy()  # Just pull out the first channel
    return data


def normalize_data(data):
    return data / np.abs(data).max()


def data_to_audio_segment(segment):
    pass


def test_midi_file_writing():
    notes = [
        Note(
            value=hz_to_midi(240 * 2**((5 * x % 12) / 12)),
            velocity=random.randint(20, 64),
            position=x / 120,
            duration=1 / 120,
        )
        for x in range(64)
        for y in range(10)
    ]
    test_file = os.path.join(DATA_DIR, "test.mid")
    create_midi_file_with_notes(
        test_file, notes
    )

    mid = mido.MidiFile(test_file, clip=True)
    track = mid.tracks[0]
    print(track)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_file")
    args = parser.parse_args()
    wav_to_midi(args.wav_file)


if __name__ == "__main__":
    main()
