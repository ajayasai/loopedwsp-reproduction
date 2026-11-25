"""Programmed suffix selection for regular, unchanged Turkish noun stems.

Each request supplies a lowercase regular stem, number and case. Two uses
of the same attention heads read the final letter and the rightmost vowel:
first from the stem, then from the stem extended by the predicted number
suffix. A linear readout chooses a suffix whose feature requirements match.
The formatter appends the two predicted suffixes to the original stem.

The rules cover singular/plural and five cases on common noun stems with
regular suffix harmony and unchanged spelling. Proper names, possessives,
exceptional harmony, vowel deletion and stem-final consonant alternation
require information beyond this model's contract. The caller supplies the
unchanged-stem assumption. Spelling validation does not verify it.

Phonology: https://yazim.tdk.gov.tr/content/02-sesler-ve-ses-uyumlari.html
Suffix inventory: https://web.itu.edu.tr/adali/wp-content/uploads/2021/10/T%C3%BCrk%C3%A7enin-%C3%87ekim-Ekleri.pdf
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
VOWELS = "aeıioöuü"
FRONT = "eiöü"
ROUNDED = "oöuü"
VOICELESS = "fstkçşhp"
NUMBERS = ("singular", "plural")
CASES = ("nominative", "locative", "ablative", "accusative", "genitive")
SUFFIXES = ("", "lar", "ler", "da", "de", "ta", "te", "dan", "den", "tan", "ten",
            "ı", "i", "u", "ü", "yı", "yi", "yu", "yü",
            "ın", "in", "un", "ün", "nın", "nin", "nun", "nün")
MAX_STEM = 24
MAX_REQUESTS = 32


def encode(record):
    requests = record.get("requests") if isinstance(record, dict) else None
    if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_REQUESTS:
        raise ValueError("Supply 1..32 requests in a list")
    out = np.zeros((len(requests), MAX_STEM + 2), dtype=np.int64)
    for i, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError("A request must supply stem, number and case")
        stem, number, case = (request.get(k) for k in ("stem", "number", "case"))
        if (not isinstance(stem, str) or not 1 <= len(stem) <= MAX_STEM or any(c not in ALPHABET for c in stem)
                or not any(c in VOWELS for c in stem)):
            raise ValueError("Use a stem of 1..24 lowercase Turkish letters containing a vowel")
        if number not in NUMBERS or case not in CASES:
            raise ValueError("Unsupported grammatical number or case")
        out[i, :len(stem)] = [ALPHABET.index(c) + 1 for c in stem]
        out[i, -2:] = NUMBERS.index(number), CASES.index(case)
    return out


def _suffix_readout():
    # Six family bits: empty, plural, locative, ablative, accusative, genitive.
    # Four agreement bits: front vowel, rounded vowel, final vowel, voiceless ending.
    weights = torch.zeros((len(SUFFIXES), 10))
    bias = torch.zeros(len(SUFFIXES))
    families = [0] + [1]*2 + [2]*4 + [3]*4 + [4]*8 + [5]*8
    for i, (suffix, family) in enumerate(zip(SUFFIXES, families)):
        weights[i, family] = 4.
        if not suffix:
            continue
        vowel = next(c for c in suffix if c in VOWELS)
        required = {6: vowel in FRONT}
        if family in (2, 3):
            required[9] = suffix.startswith("t")
        if family in (4, 5):
            required[7] = vowel in ROUNDED
            required[8] = suffix.startswith(("y", "n"))
        # Each violated agreement constraint subtracts one from the logit.
        for col, bit in required.items():
            weights[i, col] = 1. if bit else -1.
            bias[i] -= float(bit)
    return weights, bias


class InflectionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        features = [[0.]*5] + [[1., c in VOWELS, c in FRONT, c in ROUNDED, c in VOICELESS] for c in ALPHABET]
        self.orthography = nn.Parameter(torch.tensor(features), requires_grad=False)
        self.address = nn.Parameter(torch.tensor([128., 4096.]), requires_grad=False)
        weights, bias = _suffix_readout()
        self.classifier = nn.Parameter(weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        indices = [[ALPHABET.index(c) + 1 for c in suffix] for suffix in ("lar", "ler")]
        self.register_buffer("plural_spellings", torch.tensor(indices), persistent=False)

    def read(self, letters, family):
        # Heads address the last present character and last vowel. Padding
        # and ineligible keys get a penalty larger than the position range.
        position = torch.arange(letters.size(-2), dtype=letters.dtype, device=letters.device)
        eligible = letters[..., :2].transpose(-1, -2)
        scores = position * self.address[0] + (eligible - 1) * self.address[1]
        last_letter, last_vowel = (scores.softmax(-1) @ letters).unbind(-2)
        features = torch.cat((F.one_hot(family, 6).to(letters.dtype), last_vowel[..., 2:4],
                              last_letter[..., 1:2], last_letter[..., 4:5]), dim=-1)
        return F.linear(features, self.classifier, self.bias)

    def forward(self, tokens):
        letters = F.embedding(tokens[..., :MAX_STEM], self.orthography)
        number = self.read(letters, tokens[..., -2])
        # Softly emit the three letters of the inferred plural. Singular
        # contributes three padding slots; no host decision selects a suffix.
        plural = (number * self.address[0]).softmax(-1)[..., 1:3]
        spellings = self.orthography[self.plural_spellings]
        emitted = (plural @ spellings.flatten(1)).unflatten(-1, (3, 5))
        case = tokens[..., -1]
        case_family = case + (case > 0).long()
        case_logits = self.read(torch.cat((letters, emitted), dim=-2), case_family)
        return torch.stack((number, case_logits), dim=-2).flatten(1, 2)


def build_model():
    return InflectionTransformer().eval()


def reference(record):
    """Attach suffix strings in ordinary spelling order, rescanning after number."""
    encode(record)
    labels = []
    high_by_vowel = dict(zip("aıeiouöü", "ııiiuuüü"))
    for req in record["requests"]:
        stem = req["stem"]
        last = next(c for c in reversed(stem) if c in VOWELS)
        num = ""
        if req["number"] == "plural":
            num = "ler" if last in FRONT else "lar"
        word = stem + num
        last = next(c for c in reversed(word) if c in VOWELS)
        a, high = ("e" if last in FRONT else "a"), high_by_vowel[last]
        case = req["case"]
        if case in ("locative", "ablative"):
            ending = ("t" if word[-1] in VOICELESS else "d") + a + ("n" if case == "ablative" else "")
        elif case == "accusative":
            ending = ("y" if word[-1] in VOWELS else "") + high
        elif case == "genitive":
            ending = ("n" if word[-1] in VOWELS else "") + high + "n"
        else:
            ending = ""
        labels.extend((SUFFIXES.index(num), SUFFIXES.index(ending)))
    return np.asarray(labels, dtype=np.int64)


# Manually specified real-noun surface forms; each expected spelling is
# independent of the synthetic benchmark generator and reference function.
WORD_CHECKS = (
    ("ev", "plural", "nominative", "evler"), ("okul", "singular", "ablative", "okuldan"),
    ("sınıf", "singular", "ablative", "sınıftan"), ("ev", "singular", "ablative", "evden"),
    ("araba", "singular", "accusative", "arabayı"), ("kapı", "singular", "genitive", "kapının"),
    ("köprü", "singular", "genitive", "köprünün"), ("göl", "singular", "accusative", "gölü"),
    ("göl", "plural", "accusative", "gölleri"), ("yol", "singular", "genitive", "yolun"),
    ("yol", "plural", "genitive", "yolların"), ("kuş", "singular", "locative", "kuşta"),
    ("kuş", "plural", "locative", "kuşlarda"), ("süt", "singular", "locative", "sütte"),
    ("köprü", "singular", "accusative", "köprüyü"), ("gün", "singular", "nominative", "gün"),
)


def example():
    chosen = [WORD_CHECKS[i] for i in (7, 8, 11, 12, 14, 5)]
    return {"requests": [{"stem": s, "number": n, "case": c} for s, n, c, _ in chosen]}


def generate(rng, size, index):
    """Rule-test nonce stems with varying syllable count, vowel class and ending."""
    if not 1 <= size <= MAX_REQUESTS:
        raise ValueError("A batch record has 1..32 words")
    requests = []
    onsets = list("bcçdfghklmnprsştvyz")
    endings = ("", "l", "r", "n", "m", "z", "s", "ş", "f")
    for k in range(size):
        count = int(rng.integers(1, 6))
        stem = "".join(str(rng.choice(onsets)) + str(rng.choice(list(VOWELS))) for _ in range(count))
        stem += str(rng.choice(endings))
        requests.append({"stem": stem, "number": NUMBERS[(index + k) % 2],
                         "case": CASES[(index + k // 2 + int(rng.integers(len(CASES)))) % len(CASES)]})
    return {"requests": requests}


def format_output(record, prediction):
    requests = record["requests"]
    if len(prediction) != len(requests)*2:
        raise ValueError("Expected a number suffix and a case suffix for every request")
    return {"forms": [req["stem"] + SUFFIXES[int(prediction[2*i])] + SUFFIXES[int(prediction[2*i+1])]
                      for i, req in enumerate(requests)]}
