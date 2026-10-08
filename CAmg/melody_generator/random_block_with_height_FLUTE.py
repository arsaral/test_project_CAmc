# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 20:44:17 2025

@author: ars
"""

import random
from music21 import stream, note, instrument, tempo, meter

def generate_random_melody(block_base, block_height, duration, MetronomNumber, noteCount, freshnessLimit, difficulty, instrumentName, output_path):
    """
    Generates a random melody within a pitch range, ensuring no note repeats before freshnessLimit
    and avoiding jumps larger than the difficulty threshold.
    """
    base_midi = note.Note(block_base).pitch.midi
    upper_midi = base_midi + block_height

    score = stream.Score()
    part = stream.Part()
    part.id = "Melody"

    instrument_mapping = {
        "Violin": instrument.Violin(),
        "Flute": instrument.Flute(),
        "Piano": instrument.Piano(),
        "Clarinet": instrument.Clarinet(),
        "Trumpet": instrument.Trumpet(),
        "Cello": instrument.Violoncello(),
        "Bassoon": instrument.Violoncello(),
        "Oboe": instrument.Oboe()
    }
    part.append(instrument_mapping.get(instrumentName, instrument.Violin()))
    part.append(tempo.MetronomeMark(number=MetronomNumber))
    part.append(meter.TimeSignature("4/4"))

    melody = []
    recent_notes = []
    last_pitch = None

    for _ in range(noteCount):
        while True:
            available_notes = [n for n in range(base_midi, upper_midi + 1) if n not in recent_notes]
            chosen_pitch = random.choice(available_notes) if available_notes else random.randint(base_midi, upper_midi)
            
            if last_pitch is None or abs(chosen_pitch - last_pitch) < difficulty:
                break
        
        new_note = note.Note(midi=chosen_pitch, quarterLength=duration)
        melody.append(new_note)
        recent_notes.append(chosen_pitch)
        if len(recent_notes) > freshnessLimit:
            recent_notes.pop(0)
        
        last_pitch = chosen_pitch

    part.append(melody)
    score.append(part)
    score.write("musicxml", output_path)
    print(f"MusicXML file has been saved as {output_path}")

# Example usage
generate_random_melody(
    block_base="C4",        
    block_height=48,        
    duration=0.5,          
    MetronomNumber=120,     
    noteCount=120,          
    freshnessLimit=5,      
    difficulty=15,         # Maximum jump allowed in half steps
    instrumentName="Flute",
    output_path="performable_random_melody_flute.musicxml"
)
