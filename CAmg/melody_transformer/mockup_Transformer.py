# -*- coding: utf-8 -*-
# Copyright (c) 2026 Ali Rıza Saral
# Licensed under the MIT License.
"""
Created on Tue Mar 24 21:55:50 2026

@author: Ali Rıza SARAL
"""

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import random
from music21 import stream, note

# ----------------------------
# 1. Simple token system
# ----------------------------
MIN_PITCH = 50
MAX_PITCH = 80
VOCAB_SIZE = MAX_PITCH - MIN_PITCH + 1

def pitch_to_token(p):
    return p - MIN_PITCH

def token_to_pitch(t):
    return t + MIN_PITCH


# ----------------------------
# 2. Tiny Transformer
# ----------------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size + 10, d_model)
        self.pos = nn.Parameter(torch.randn(1, 512, d_model))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos[:, :seq_len, :]
        x = self.transformer(x)
        return self.fc(x)


# ----------------------------
# 3. Generate base random cloud
# ----------------------------
def generate_random_sequence(length=40):
    seq = []
    for _ in range(length):
        pitch = random.randint(60, 65)  # narrow cloud
        seq.append(pitch_to_token(pitch))
    return seq


# ----------------------------
# 4. Add dynamic control
# ----------------------------
def add_time_control(seq):
    new_seq = []
    for i, t in enumerate(seq):
        # control signal: expanding range over time
        control = int(5 * (i / len(seq)))  # grows from 0 → 5
        new_seq.append(t + control)
    return new_seq


# ----------------------------
# 5. "Fake-trained" behavior (important trick)
# ----------------------------
def simulate_transformer_behavior(seq):
    """
    Instead of real training, we simulate a learned transformation:
    - early: narrow
    - middle: expanding
    - late: wide & jumpy
    """

    new_seq = []

    for i, t in enumerate(seq):
        progress = i / len(seq)

        if progress < 0.3:
            # smooth region
            delta = random.choice([-1, 0, 1])
        elif progress < 0.7:
            # expanding
            delta = random.choice([-3, -2, 0, 2, 3])
        else:
            # wide jumps
            delta = random.choice([-7, -5, 0, 5, 7])

        new_token = max(0, min(VOCAB_SIZE - 1, t + delta))
        new_seq.append(new_token)

    return new_seq


# ----------------------------
# 6. Convert to music21
# ----------------------------
def sequence_to_stream(seq):
    s = stream.Stream()

    for t in seq:
        p = token_to_pitch(t)
        n = note.Note(p)
        n.quarterLength = 0.5
        s.append(n)

    return s


# ----------------------------
# 7. MAIN
# ----------------------------
# Step 1: base cloud
base_seq = generate_random_sequence()

# Step 2: dynamic shaping (Transformer idea)
dynamic_seq = simulate_transformer_behavior(base_seq)

# Step 3: convert to score
score = sequence_to_stream(dynamic_seq)

# Step 4: write to .mxl file
output_path = score.write('musicxml', 'dynamic_cloud.mxl')

print("Saved to:", output_path)