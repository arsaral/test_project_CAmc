# -*- coding: utf-8 -*-
# Copyright (c) 2026 Ali Rıza Saral
# Licensed under the MIT License.
"""
@author: Ali Rıza SARAL

Improved Transformer Melody Model
Fixes applied:
- Preserve interval sign
- Remove sorting
- Remove pre-scaling in inference
- Enforce output length
- Use round instead of int
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

# --- Configuration ---
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

VOCAB_SIZE = 200
MAX_LEN = 20
SHIFT = 50

DEVICE = torch.device("cpu")  # safer


# =========================
# TRANSFORMATIONS
# =========================
def expand_melody(melody, low, high):
    out = [melody[0]]
    for i in range(1, len(melody)):
        interval = melody[i] - melody[i - 1]
        new_note = out[-1] + round(interval * 1.5)
        new_note = max(low, min(high, new_note))
        out.append(new_note)
    return out


def reduce_melody(melody, low, high):
    out = [melody[0]]
    for i in range(1, len(melody)):
        interval = melody[i] - melody[i - 1]
        new_note = out[-1] + round(interval * 0.5)
        new_note = max(low, min(high, new_note))
        out.append(new_note)
    return out


def apply_transformation(melody, t_type, low, high):
    if t_type == 0:
        return expand_melody(melody, low, high)
    else:
        return reduce_melody(melody, low, high)


# =========================
# DATA GENERATION
# =========================
def generate_pair():
    length = random.randint(5, 10)
    low = random.randint(40, 60)
    high = low + random.randint(12, 24)

    melody = [random.randint(low, high) for _ in range(length)]
    t_type = random.choice([0, 1])

    transformed = apply_transformation(melody, t_type, low, high)

    # intervals (SIGNED)
    src_intervals = [melody[i] - melody[i - 1] for i in range(1, len(melody))]
    tgt_intervals = [transformed[i] - transformed[i - 1] for i in range(1, len(transformed))]

    def clip_shift(interval):
        interval = max(-20, min(20, interval))
        return interval + SHIFT

    src_intervals = [clip_shift(i) for i in src_intervals]
    tgt_intervals = [clip_shift(i) for i in tgt_intervals]

    # control tokens
    T_TOKEN = 80 + t_type * 30
    LOW_TOKEN = 10 + (low - 30)
    HIGH_TOKEN = 40 + (high - 30)
    control = [T_TOKEN, LOW_TOKEN, HIGH_TOKEN]

    src_seq = control + src_intervals
    tgt_seq = control + tgt_intervals

    return src_seq, tgt_seq, melody[0], len(src_intervals)


def pad_sequence(seq, max_len):
    return seq + [PAD_IDX] * (max_len - len(seq))


def prepare_batch(batch_size=32):
    src_batch, tgt_batch, lengths = [], [], []

    for _ in range(batch_size):
        src, tgt, _, length = generate_pair()

        src = [SOS_IDX] + src + [EOS_IDX]
        tgt = [SOS_IDX] + tgt + [EOS_IDX]

        src = pad_sequence(src[:MAX_LEN], MAX_LEN)
        tgt = pad_sequence(tgt[:MAX_LEN], MAX_LEN)

        src_batch.append(src)
        tgt_batch.append(tgt)
        lengths.append(length)

    return (
        torch.tensor(src_batch, dtype=torch.long, device=DEVICE),
        torch.tensor(tgt_batch, dtype=torch.long, device=DEVICE),
        lengths
    )


# =========================
# POSITIONAL ENCODING
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


# =========================
# MODEL
# =========================
class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)).to(tgt.device)
        src_emb = self.pos_enc(self.embedding(src) * (self.d_model ** 0.5))
        tgt_emb = self.pos_enc(self.embedding(tgt) * (self.d_model ** 0.5))
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(out)


# =========================
# TRAIN
# =========================
model = Seq2SeqTransformer(VOCAB_SIZE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

for epoch in range(2000):
    model.train()
    src, tgt, _ = prepare_batch(64)
    optimizer.zero_grad()
    output = model(src, tgt[:, :-1])
    loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


# =========================
# INFERENCE
# =========================
def translate(input_melody, t_type, low, high):
    model.eval()

    intervals = [input_melody[i] - input_melody[i - 1] for i in range(1, len(input_melody))]

    def clip_shift(interval):
        interval = max(-20, min(20, interval))
        return interval + SHIFT

    intervals = [clip_shift(i) for i in intervals]

    T_TOKEN = 80 + t_type * 30
    LOW_TOKEN = 10 + (low - 30)
    HIGH_TOKEN = 40 + (high - 30)

    src_seq = [SOS_IDX] + [T_TOKEN, LOW_TOKEN, HIGH_TOKEN] + intervals + [EOS_IDX]
    src_seq = pad_sequence(src_seq, MAX_LEN)
    src = torch.tensor([src_seq], dtype=torch.long, device=DEVICE)

    tgt_seq = [SOS_IDX, T_TOKEN, LOW_TOKEN, HIGH_TOKEN]

    expected_len = len(input_melody) - 1

    for _ in range(expected_len):
        tgt_padded = pad_sequence(tgt_seq, MAX_LEN)
        tgt_tensor = torch.tensor([tgt_padded], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            output = model(src, tgt_tensor)

        logits = output[0, len(tgt_seq)-1]
        logits[EOS_IDX] -= 5.0
        next_token = logits.argmax().item()

        tgt_seq.append(next_token)

    intervals_out = [t - SHIFT for t in tgt_seq[4:]]

    melody_out = [input_melody[0]]
    for i in intervals_out:
        melody_out.append(melody_out[-1] + i)

    return melody_out


# --- TEST ---
test_input = [55, 65, 60, 62, 72, 65, 70, 64, 75]

expanded_output = translate(test_input, 0, 50, 75)
reduced_output = translate(test_input, 1, 50, 85)

print("\n--- INPUT ---")
print(test_input)

print("\n--- EXPAND ---")
print(expanded_output)

print("\n--- REDUCE ---")
print(reduced_output)

# =========================
# PLOT
# =========================
import matplotlib
matplotlib.use('Agg')  # SAFE backend (no GUI crash)
import matplotlib.pyplot as plt

plt.figure()

x_input = list(range(len(test_input)))
x_expanded = list(range(len(expanded_output)))
x_reduced = list(range(len(reduced_output)))

plt.plot(x_input, test_input, marker='o', label="Input")
plt.plot(x_expanded, expanded_output, marker='o', linestyle='--', label="Expanded")
plt.plot(x_reduced, reduced_output, marker='o', linestyle=':', label="Reduced")

plt.xlabel("Time Step")
plt.ylabel("Pitch")
plt.title("Melody Transformation")
plt.legend()

# Save instead of show
plt.savefig("melody_plot.png")
plt.close()

"""
python transformer41.py
Epoch 0: Loss = 5.4714
Epoch 100: Loss = 2.1960
Epoch 200: Loss = 1.6144
Epoch 300: Loss = 1.6958
Epoch 400: Loss = 1.4592
Epoch 500: Loss = 1.4317
Epoch 600: Loss = 1.4338
Epoch 700: Loss = 1.1338
Epoch 800: Loss = 1.1912
Epoch 900: Loss = 0.9977
Epoch 1000: Loss = 1.0857
Epoch 1100: Loss = 0.9149
Epoch 1200: Loss = 0.9375
Epoch 1300: Loss = 0.9206
Epoch 1400: Loss = 0.8195
Epoch 1500: Loss = 0.8463
Epoch 1600: Loss = 0.7117
Epoch 1700: Loss = 0.7081
Epoch 1800: Loss = 0.6917
Epoch 1900: Loss = 0.7054

--- INPUT ---
[55, 65, 60, 62, 72, 65, 70, 64, 75]

--- EXPAND ---
[55, 70, 62, 65, 74, 64, 76, 67, 79]

--- REDUCE ---
[55, 60, 58, 59, 55, 57, 63, 60, 66]

"""