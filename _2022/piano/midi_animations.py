from manim_imports_ext import *

import mido
from _2022.piano.wav_to_midi import DATA_DIR
from _2022.piano.wav_to_midi import piano_midi_range


class AnimatedMidi(Scene):
    midi_file = "3-16-attempts/Help_Long_as_piano_5ms.mid"
    dist_per_sec = 4.0
    bpm = 240
    CONFIG = {
        "camera_config": {
            "anti_alias_width": 0,
            "samples": 4,
        }
    }
    hit_depth = 0.025
    note_color = BLUE
    hit_color = TEAL

    def construct(self):
        self.add_piano()
        self.add_note_rects()
        self.scroll()

    def add_piano(self):
        piano = Piano3D()
        piano.set_width(9)
        piano.move_to(5 * DOWN)

        frame = self.camera.frame
        frame.set_phi(70 * DEGREES)

        self.piano = piano
        self.add(piano)

    def add_note_rects(self):
        mid_file = os.path.join(DATA_DIR, self.midi_file)
        mid = mido.MidiFile(mid_file, clip=True)
        track = mid.tracks[0]
        offset = piano_midi_range[0]
        sec_per_tick = (60.0 / self.bpm) / mid.ticks_per_beat
        dist_per_tick = self.dist_per_sec * sec_per_tick

        pending_notes = {key: None for key in piano_midi_range}
        notes_to_time_spans = {key: [] for key in piano_midi_range}

        note_rects = VGroup()
        time_in_ticks = 0
        for msg in track:
            time_in_ticks += msg.time
            if msg.type == 'note_on':
                pending_notes[msg.note] = (time_in_ticks, msg.velocity)
            elif msg.type == 'note_off':
                if msg.note not in pending_notes:
                    continue
                start_time_in_ticks, velocity = pending_notes.pop(msg.note)
                key = self.piano[msg.note - offset]
                rect = Rectangle(
                    width=key.get_width(),
                    height=(time_in_ticks - start_time_in_ticks) * dist_per_tick
                )
                rect.next_to(key, UP, buff=start_time_in_ticks * dist_per_tick)
                rect.set_stroke(width=0)
                rect.set_fill(self.note_color, opacity=clip(0.25 + velocity / 100, 0, 1))
                note_rects.add(rect)
                notes_to_time_spans[msg.note].append((
                    start_time_in_ticks * sec_per_tick,
                    time_in_ticks * sec_per_tick
                ))

        self.note_rects = note_rects
        self.notes_to_time_spans = notes_to_time_spans
        self.add(note_rects)

    def scroll(self):
        piano = self.piano
        note_rects = self.note_rects
        notes_to_time_spans = self.notes_to_time_spans
        note_rects.add_updater(lambda m, dt: m.shift(self.dist_per_sec * dt * DOWN))

        for key in piano:
            key.original_z = key.get_z()
            key.original_color = key[0].get_fill_color()

        piano.time = 0

        def update_piano(piano, dt):
            piano.time += dt
            for note, key in zip(piano_midi_range, piano):
                hit = False
                for start, end in notes_to_time_spans[note]:
                    if start - 1 / 60 < piano.time < end + 1 / 60:
                        hit = True
                if hit:
                    key.set_z(key.original_z - self.hit_depth)
                    key.set_fill(interpolate_color(
                        key[0].get_fill_color(), self.hit_color, 0.5
                    ))
                else:
                    key.set_z(key.original_z)
                    key.set_fill(key.original_color)

        piano.add_updater(update_piano)

        self.add_sound(
            os.path.join(DATA_DIR, self.midi_file).replace(".mid", ".wav"),
            time_offset=0.3,
        )
        self.wait(note_rects.get_height() / self.dist_per_sec + 3.0)
