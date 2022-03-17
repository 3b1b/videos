from manim_imports_ext import *

import argparse
import matplotlib.pyplot as plt

import mido
from collections import namedtuple
from tqdm import tqdm as ProgressDisplay

from scipy.signal import fftconvolve
from scipy.signal import convolve
from scipy.signal import argrelextrema
from scipy.io import wavfile

from IPython.terminal.embed import InteractiveShellEmbed
embed = InteractiveShellEmbed()


SAMPLED_VELOCITY = 100
SAMPLED_VELOCITIES = list(range(25, 150, 25))
DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)

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


def projection_factor(v, w):
    """
    If projecting v onto w produces the vector f * w, this returns f
    """
    return np.dot(v, w) / np.dot(w, w)


def gaussian_kernel(length=100, spread=0.5):
    """
    creates gaussian kernel with side length `l` and a sigma of `sigma`
    """
    sigma = spread * length
    ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    return gauss / gauss.sum()


# Functions for creating MIDI files


def hz_to_midi_value(frequencies):
    freqs = np.atleast_1d(frequencies)
    return (12 * np.log2(freqs / 440) + 69).astype(int)


def midi_value_to_hz(midis):
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
        sec_per_tick = (60.0 / bpm) / midifile.ticks_per_beat

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
    wav_file = mid_file.replace(".mid", ".wav")
    mp3_file = mid_file.replace(".mid", ".mp3")
    if os.path.exists(wav_file):
        os.remove(wav_file)
    os.system(" ".join([
        "timidity",
        mid_file,
        "-Ow -o -",
        "|",
        "ffmpeg",
        "-i - -acodec libmp3lame -ab 64k",
        "-ar 48000",
        "-hide_banner -loglevel error",
        mp3_file,
    ]))
    os.system(" ".join([
        "ffmpeg",
        "-hide_banner -loglevel error",
        "-i",
        mp3_file,
        wav_file,
    ]))
    os.remove(mp3_file)
    return wav_file


def generate_pure_piano_key_files(velocities=[SAMPLED_VELOCITY], duration=0.025, folder="digital_piano_samples"):
    folder = os.path.join(DATA_DIR, folder)
    if not os.path.exists(folder):
        os.makedirs(folder)

    for key in piano_midi_range:
        for vel in velocities:
            note = Note(key, vel, 0, duration)
            mid_file = os.path.join(folder, f"{key}_{vel}.mid")
            create_midi_file_with_notes(mid_file, [note])
            midi_to_wav(mid_file)
            os.remove(mid_file)


def generate_piano_sample_midi(delay=5, duration=0.025, velocity=100):
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


# Processing sample signals from piano

def load_piano_key_signals(folder="piano_samples", duration=0.5, velocity=50):
    key_signals = []
    for key in piano_midi_range:
        sample_rate, full_signal = wav_data(
            os.path.join(DATA_DIR, folder, f"{key}_{velocity}.wav")
        )
        key_signal = full_signal[:int(duration * sample_rate)]
        key_signals.append(key_signal)
    return np.array(key_signals, dtype=float)


def get_volume_to_velocity_func_map(folder="true_piano_samples", sampled_velocities=[26, 101]):
    """
    Functions are encoded as a pair (c0, c1) for the linear function c0 + c1 * x
    """
    result = dict()
    vels = sampled_velocities
    for key in piano_midi_range:
        globals().update(locals())
        volumes = [
            wav_data(os.path.join(DATA_DIR, folder, f"{key}_{vel}.wav"))[1].max()
            for vel in vels
        ]
        # Coefficients for const + (slope) * x function fit
        slope = (vels[1] - vels[0]) / (volumes[1] - volumes[0])
        const = vels[0] - slope * volumes[0]
        result[key] = (const, slope)
    return result


def get_velocity(key, signal, scale_factor, v2v_func_map):
    c0, c1 = v2v_func_map[key]
    return c0 + c1 * (signal.max() * scale_factor)


def shift_pitch(signal, sample_rate, shift_in_hz, frame_size=4800):
    result = []
    for lh in range(0, len(signal), frame_size):
        rh = lh + frame_size
        fft_spectrum = np.fft.rfft(signal[lh:rh])
        freq = sample_rate / (rh - lh)
        shift = int(shift_in_hz / freq)
        shifted_spectrum = np.zeros_like(fft_spectrum)
        shifted_spectrum[shift:] = fft_spectrum[:-shift]
        shifted_signal = np.fft.irfft(shifted_spectrum)
        result.append(shifted_signal)
    return np.hstack(result)


def wav_to_midi(sound_file,
                output_file=None,
                key_block_time=0.075,
                key_signal_time=0.075,
                key_play_time=0.020,
                step_size=0.001,
                sample_velocity=100,
                sample_folder="digital_piano_samples",
                key_signal_max=0.15,  # This has a very large effect
                volume_ratio_threshold=1.0,
                scale_factor_cutoff=0.1,
                min_velocity=5,
                max_velocity=60,
                # Honestly, low keys are trash, so many are supressed
                n_supressed_lower_keys=32,
                supression_factor=0.25,
                ):
    """
    Walk through a series of windows over the original signal, and for each one,
    find the top several key sounds which correlate most closely with that window.
    More specifically, do a convolution to let that piano key signal 'slide' along
    the window to find the best possible match.

    Room for improvement:
        - Duration shouldn't necessarily be fixed
    """
    # Read in audio file, normalize
    sample_rate, signal = wav_data(sound_file)
    signal /= signal.max()
    new_signal = np.zeros_like(signal)

    # We (potentially) add one note per step_size. A single note cannot be played
    # multiple times within a key_block_size window.
    step_size = int(sample_rate * step_size)
    key_block_size = int(sample_rate * key_block_time)

    notes = []
    key_signals = load_piano_key_signals(
        folder=sample_folder,
        duration=key_signal_time,
        velocity=sample_velocity,
    )
    key_signals *= key_signal_max / key_signals.max()

    velocities = []  # Just for debugging, can delete

    # Compute correlations between all notes at all points along the signal
    full_convs = np.array([
        fftconvolve(signal, ks[::-1])[len(ks) - 1:]
        for ks in key_signals
    ])
    # Repress lower keys.  TODO: Instead of multiplying by an arbitrary factor, use some smoothing function
    full_convs[:n_supressed_lower_keys, :] *= supression_factor

    # The signal is divided into many little windows, with these "positions" marking
    # the first index of each such window. These are sorted based on which ones
    # have the highest correlation with some particular key.
    positions = list(range(0, len(signal), step_size))
    positions.sort(key=lambda p: -full_convs[:, p:p + step_size].max())
    for pos in positions:
        window = signal[pos:pos + step_size]
        new_window = new_signal[pos:pos + step_size]

        # When volume is larger than original window, stop adding new notes
        if norm(new_window) > volume_ratio_threshold * norm(window):
            continue

        convs = full_convs[:, pos:pos + step_size]
        key_index, offset = np.unravel_index(np.argmax(convs), convs.shape)
        opt_pos = pos + offset
        ks = key_signals[key_index]

        # Consider the segment of the original signal which lines up
        # with this key signal as a vector. If you project that segment
        # onto the key signal itself, producing a vector which is f * (key signal),
        # this factor gives f.
        diff = (signal - new_signal)[opt_pos:opt_pos + len(ks)]  # What's the right length here?
        factor = projection_factor(diff, ks[:len(diff)])

        if factor > scale_factor_cutoff:
            velocity = int(interpolate(min_velocity, max_velocity, clip(factor, 0, 1)))
            velocities.append((factor, velocity))
            # Add the key_signal to new_window, which will in turn be added to new_signal
            piece = new_signal[opt_pos:opt_pos + len(ks)]
            piece += (velocity / sample_velocity) * ks[:len(piece)]
            # Mark this key as unavailable for the next len(ks) samples
            full_convs[key_index, opt_pos:opt_pos + key_block_size] = 0
            # Add the note, which will ultimately be used to create the MIDI file
            notes.append(Note(
                value=piano_midi_range[key_index],
                velocity=velocity,
                position=opt_pos / sample_rate,
                # Always hit with a short staccato
                duration=key_play_time,
            ))

    if output_file is None:
        output_file = sound_file.replace(".wav", "_as_piano.mid")
    create_midi_file_with_notes(output_file, notes)
    midi_to_wav(output_file)

    # plt.plot(signal, linewidth=1.0)
    # plt.plot(new_signal, linewidth=1.0)
    # plt.plot(smooth_signal, linewidth=1.0)
    # plt.show()
    return


# Functions for processing sound files


def show_down_midi_file(mid_file, slow_factor=4):
    mid = mido.MidiFile(mid_file, clip=True)
    track = mid.tracks[0]

    for msg in track:
        if msg.type in ['note_on', 'note_off']:
            msg.time *= slow_factor
    new_file = mid_file.replace(".mid", f"_slowed_by_{slow_factor}.mid")
    mid.save(new_file)
    return new_file


def wav_data(file_name):
    rate, arr = wavfile.read(file_name)
    arr = arr.astype(float)
    if len(arr.shape) > 1 and arr.shape[1] > 1:
        arr = arr.mean(1)  # Collapse to single channel
    return rate, arr


def extract_pure_keys(key_sample_file,
                      output_folder="true_piano_samples",
                      start=3.9,
                      spacing=3,
                      velocities=[1, 26, 51, 76, 101, 126]):
    """
    For a file
    """
    output_folder = os.path.join(DATA_DIR, output_folder)
    sample_rate, arr = wav_data(key_sample_file)

    # Create function whose local maxima correspond to key starts
    kernel = np.ones(sample_rate) / sample_rate
    smoothed = fftconvolve(np.abs(arr).astype(float), kernel, mode='valid')
    smoothed = fftconvolve(smoothed, kernel, mode='valid')
    smooth_to_true_shift = int(0.7 * sample_rate)
    smoothed = np.hstack([np.zeros(smooth_to_true_shift), smoothed])
    local_maxima = argrelextrema(smoothed, np.greater)[0]

    def true_peak_near(index):
        return local_maxima[np.argmin(np.abs(local_maxima - index))]

    spacing_in_samples = int(spacing * sample_rate)
    index = true_peak_near(start * sample_rate)
    for key in piano_midi_range:
        for vel in velocities:
            wavfile.write(
                filename=os.path.join(output_folder, f"{key}_{vel}.wav"),
                rate=sample_rate,
                data=arr[index:index + spacing_in_samples],
            )
            index = true_peak_near(index + spacing_in_samples)
    return


def normalize_data(data):
    return data / np.abs(data).max()


def data_to_audio_segment(segment):
    pass


def test_midi_file_writing():
    notes = [
        Note(
            value=hz_to_midi_value(240 * 2**((5 * x % 12) / 12)),
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


def plot_velocity_data():
    key = 55
    folder = os.path.join(DATA_DIR, "true_piano_samples")
    signals = []
    vels = list(range(1, 127, 25))
    for vel in vels:
        file = os.path.join(folder, f"{key}_{vel}.wav")
        rate, signal = wav_data(file)
        signals.append(signal)

    maxes = [s.max() for s in signals]
    plt.plot(maxes, vels)
    plt.plot(np.linspace(0, maxes[-1], len(maxes)), vels)
    plt.show()

    embed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_file")
    parser.add_argument("output_file", nargs="?")
    args = parser.parse_args()
    wav_to_midi(args.wav_file, args.output_file)


if __name__ == "__main__":
    main()


    # mid1 = mido.MidiFile("/Users/grant/cs/videos/_2022/piano/data/better_midis/TrappedInAPiano.wav.mid")
    # mid2 = mido.MidiFile("/Users/grant/cs/videos/_2022/piano/data/3-9-attempts/DensityTest_5ms.mid")
    # track1 = mid1.tracks[0]
    # track2 = mid2.tracks[0]