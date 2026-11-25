"""Repair eight-track frames of a synthetic, parity-protected punched music roll.

Frame strings list observed holes in track order 1..8. Data tracks 3, 5, 6,
and 7 store a four-bit event ID, least significant bit first. Tracks 1, 2,
and 4 hold even Hamming parity, and track 8 gives even parity over all eight
tracks. The event alphabet is a rest followed by the C-major notes C4..C6.
Each frame represents one beat. The guarantee assumes at most two damaged
holes in a frame. Two errors make the original note ambiguous.

The tied transformer applies positional attention to four check channels
through a three-level binary reduction. Its ReLU feedforward computes XOR
from the attended mean of each adjacent pair. A linear head scores the
joint agreement of the resulting four parity bits and four observed data
bits with each legal note/repair combination. It includes a rejection class
for syndromes that cannot describe a clean or singly damaged frame.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


NOTES = ("rest", "C4", "D4", "E4", "F4", "G4", "A4", "B4",
         "C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6")
DATA_TRACKS = (3, 5, 6, 7)
UNCORRECTABLE = 144
MAX_FRAMES = 128


def encode(record):
    """Convert literal hole strings to [frames,8] raw bit tokens."""
    frames = record.get("frames") if isinstance(record, dict) else None
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError(f"Provide a frames list of length 1..{MAX_FRAMES}")
    if any(not isinstance(row, str) or len(row) != 8 or not set(row) <= {"0", "1"} for row in frames):
        raise ValueError("Each frame must be a string of eight binary digits in track order")
    return np.array([[int(bit) for bit in row] for row in frames], dtype=np.int64)


class ParityBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_query = nn.Parameter(torch.tensor([64., -32.]), requires_grad=False)
        self.parity_response = nn.Parameter(torch.tensor([2., -4., -0.5]), requires_grad=False)

    def forward(self, state):
        slots = state.shape[-2]
        position = torch.arange(slots, dtype=state.dtype, device=state.device)
        # A query midway between siblings has equal scores at the two siblings.
        # The next-nearest track receives at most exp(-64) of a sibling's mass.
        center = 2 * position[:slots//2] + 0.5
        q = torch.stack((center, torch.ones_like(center)), dim=-1) * self.position_query
        k = torch.stack((position, position.square()), dim=-1)
        mean = (q @ k.T).softmax(dim=-1) @ state
        linear, hinge, offset = self.parity_response
        return linear * mean + hinge * F.relu(mean + offset)


class RollTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # These fixed projection weights select the contributing tracks for
        # the three Hamming checks and the fourth overall parity check.
        membership = [[int(track < 8 and bool(track & mask)) for mask in (1, 2, 4)] + [1]
                      for track in range(1, 9)]
        self.membership = nn.Parameter(torch.tensor(membership, dtype=torch.float32), requires_grad=False)
        self.block = ParityBlock()
        # Track zero denotes a clean frame. A corrected track changes one
        # observed data bit precisely when the damage is on a data track.
        readout = []
        for track in range(9):
            syndrome = [int(track < 8 and bool(track & mask)) for mask in (1, 2, 4)] + [int(track > 0)]
            for event in range(16):
                observed = [((event >> j) & 1) ^ int(track == place) for j, place in enumerate(DATA_TRACKS)]
                expected = syndrome + observed
                # Dot with [features, 1] to count matching bits out of eight.
                readout.append([2*bit - 1 for bit in expected] + [expected.count(0)])
        readout.append([0]*8 + [7.5])
        self.readout = nn.Parameter(torch.tensor(readout, dtype=torch.float32), requires_grad=False)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != 8 or not 1 <= tokens.shape[1] <= MAX_FRAMES:
            raise ValueError("Expected [batch, 1..128 frames, 8 tracks] raw bit tokens")
        bits = tokens.to(dtype=self.membership.dtype)
        state = bits.unsqueeze(-1) * self.membership
        for _ in range(3):
            state = self.block(state)
        observed = bits[..., [t-1 for t in DATA_TRACKS]]
        features = torch.cat((state.squeeze(-2), observed), dim=-1)
        return F.pad(features, (0, 1), value=1.) @ self.readout.T


def build_model():
    return RollTransformer().eval()


def _word(event):
    """Encode a pristine event by the four independent even-parity constraints."""
    a, b, c, d = [(event >> j) & 1 for j in range(4)]
    first_seven = [a ^ b ^ d, a ^ c ^ d, a, b ^ c ^ d, b, c, d]
    return first_seven + [sum(first_seven) % 2]


def reference(record):
    """Resolve by Hamming distance from each pristine codeword, independent of the block."""
    received = encode(record)
    codebook = np.array([_word(i) for i in range(16)])
    different = received[:, None, :] != codebook[None, :, :]
    distance = different.sum(axis=-1)
    answer = []
    for mask, distances in zip(different, distance):
        event = int(distances.argmin())
        if distances[event] == 0:
            answer.append(event)
        elif distances[event] == 1:
            track = int(np.flatnonzero(mask[event])[0]) + 1
            answer.append(16*track + event)
        else:
            answer.append(UNCORRECTABLE)
    return np.array(answer, dtype=np.int64)


def _punch(event, flipped):
    bits = _word(event)
    for track in flipped:
        bits[track - 1] ^= 1
    return "".join(map(str, bits))


def example():
    # A clean middle C, a D needing track 5 repaired, an ambiguous frame, then a rest.
    return {"frames": [_punch(1, []), _punch(2, [5]), _punch(3, [1, 7]), _punch(0, [])]}


def generate(rng, size, index):
    if size not in (4, 8, 16, 32):
        raise ValueError("The benchmark uses 4, 8, 16 or 32 frames")
    # Every generated roll has clean, correctable and uncorrectable frames,
    # distributed randomly across the roll. The original events stay private
    # to the generator; no damage counts or locations accompany a record.
    damage = rng.choice(3, size=size, p=[0.4, 0.4, 0.2])
    damage[:3] = (0, 1, 2)
    rng.shuffle(damage)
    frames = []
    for count in damage:
        event = int(rng.integers(16))
        flipped = 1 + rng.choice(8, size=int(count), replace=False)
        frames.append(_punch(event, flipped))
    return {"frames": frames}


def format_output(record, prediction):
    frames = encode(record)
    if len(prediction) != len(frames):
        raise ValueError("Expected one predicted class per frame")
    events = []
    for value in prediction:
        if not isinstance(value, (int, np.integer)) or not 0 <= value <= UNCORRECTABLE:
            raise ValueError("Invalid roll class")
        if value == UNCORRECTABLE:
            events.append({"note": None, "status": "double_error"})
        else:
            track, event = divmod(int(value), 16)
            if track == 0:
                events.append({"note": NOTES[event], "status": "clean"})
            else:
                events.append({"note": NOTES[event], "status": "corrected", "track": track})
    return {"events": events}
