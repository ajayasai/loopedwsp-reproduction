"""Resolve paired Tzolk'in and Haab' designations within one Maya Calendar Round."""
import numpy as np
import torch
from torch import nn


NAMES = ("Imix", "Ik'", "Akb'al", "K'an", "Chicchan", "Kimi", "Manik'", "Lamat", "Muluk", "Ok",
         "Chuwen", "Eb'", "B'en", "Ix", "Men", "Kib'", "Kab'an", "Etz'nab'", "Kawak", "Ajaw")
MONTHS = ("Pop", "Wo", "Sip", "Sotz'", "Sek", "Xul", "Yaxk'in", "Mol", "Ch'en", "Yax", "Sak",
          "Kej", "Mak", "K'ank'in", "Muwan", "Pax", "K'ayab'", "Kumk'u", "Wayeb'")
ROUND = 18980
NO_DATE = 365


def encode(record):
    if not isinstance(record, dict) or set(record) != {"dates"}:
        raise ValueError("Supply one dates list")
    dates = record["dates"]
    if not isinstance(dates, list) or not 1 <= len(dates) <= 64:
        raise ValueError("Supply 1–64 Calendar Round designations")
    tokens = []
    for item in dates:
        if not isinstance(item, dict) or set(item) != {"number", "name", "day", "month"}:
            raise ValueError("Each date requires number, name, day and month")
        number, name = item["number"], item["name"]
        day, month = item["day"], item["month"]
        if type(number) is not int or not 1 <= number <= 13 or name not in NAMES:
            raise ValueError("Supply a Tzolk'in number from 1–13 and a listed day name")
        if month not in MONTHS or type(day) is not int or not 0 <= day < (5 if month == "Wayeb'" else 20):
            raise ValueError("Use an ordinal of 0–4 in Wayeb' and 0–19 in the other listed months")
        tokens.append((number, NAMES.index(name), day, MONTHS.index(month)))
    return np.asarray(tokens, dtype=np.int64)


def _calendar_day(offset):
    haab = (348 + offset) % 365
    return {"number": 1 + (3 + offset) % 13, "name": NAMES[(19 + offset) % 20],
            "day": haab % 20, "month": MONTHS[haab // 20]}


class CalendarRoundTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # A pair of integral coordinates addresses a day of the 260-day cycle.
        days = torch.arange(260, dtype=torch.float32)
        numbers, names = 1 + (3 + days) % 13, (19 + days) % 20
        memory = torch.stack((numbers, names, numbers.square() + names.square(), days), dim=1)
        self.tzolkin = nn.Parameter(memory, requires_grad=False)
        self.haab = nn.Parameter(torch.tensor([1., 20., -348.]), requires_grad=False)
        self.sharpness = nn.Parameter(torch.tensor(64.), requires_grad=False)

        # Difference = tzolkin offset - haab offset. A matching key gives
        # the number of 365-day intervals after the declared origin.
        entries = []
        for year in range(52):
            residue = (365 * year) % 260
            entries.extend((key, year) for key in (residue-520, residue-260, residue) if -364 <= key <= 259)
        self.year_memory = nn.Parameter(torch.tensor(entries, dtype=torch.float32), requires_grad=False)

    def forward(self, tokens):
        raw = tokens.to(self.tzolkin.dtype)
        number, name, day, month = raw.unbind(-1)
        one = torch.ones_like(number)
        queries = torch.stack((2*number, 2*name, -one), dim=-1)
        # Dot products differ from negative squared distances by a query constant.
        proximity = queries @ self.tzolkin[:, :3].T
        tz_day = ((self.sharpness * proximity).softmax(-1) @ self.tzolkin[:, 3:]).squeeze(-1)
        haab_day = torch.stack((day, month, one), dim=-1) @ self.haab
        haab_day = haab_day + 365 * (-haab_day).clamp(0, 1)

        difference = tz_day - haab_day
        address, year_ids = self.year_memory.unbind(1)
        keys = torch.stack((address, -address.square()))
        queries = torch.stack((2*difference, one), dim=-1)
        distance_scores = queries @ keys - difference.square().unsqueeze(-1)
        # Half an integer-distance unit: a perfect match beats the fallback,
        # and every miss loses to it. Its value is the incompatible class.
        fallback = one.unsqueeze(-1) * -0.5
        weights = (self.sharpness * torch.cat((distance_scores, fallback), dim=-1)).softmax(-1)
        no_match = weights[..., -1]
        year = weights[..., :-1] @ year_ids + NO_DATE * no_match
        ordinal = haab_day * (1 - no_match) + NO_DATE * no_match
        answers = torch.stack((year, ordinal), dim=-1).flatten(1, 2)
        labels = torch.arange(NO_DATE + 1, dtype=raw.dtype, device=raw.device)
        return -(answers.unsqueeze(-1) - labels).abs()


def build_model():
    return CalendarRoundTransformer().eval()


def reference(record):
    # Generate all 52 occurrences of the requested Haab' date. Keep the one
    # which also has the requested coefficient and name, if it exists.
    result = []
    for number, name, day, month in encode(record).tolist():
        ordinal = (20*month + day - 348) % 365
        for year in range(52):
            offset = 365*year + ordinal
            if 1 + (3+offset) % 13 == number and (19+offset) % 20 == name:
                result.extend((year, ordinal))
                break
        else:
            result.extend((NO_DATE, NO_DATE))
    return np.asarray(result, dtype=np.int64)


def generate(rng, size, index):
    if not 1 <= size <= 64:
        raise ValueError("Expected 1–64 requests")
    dates = []
    while len(dates) < size:
        # Reserve some examples for the five-day final month of the Haab'.
        if (index + len(dates)//2) % 4 == 0:
            offset = int(rng.integers(52))*365 + 12 + int(rng.integers(5))
        else:
            offset = int(rng.integers(ROUND))
        valid = _calendar_day(offset)
        dates.append(valid)
        if len(dates) < size:
            near_miss = valid.copy()
            name_id = (NAMES.index(valid["name"]) + int(rng.integers(1, 5))) % 20
            near_miss["name"] = NAMES[name_id]
            dates.append(near_miss)
    rng.shuffle(dates)
    return {"dates": dates}


def example():
    return {"dates": [
        {"number": 4, "name": "Ajaw", "day": 8, "month": "Kumk'u"},
        {"number": 4, "name": "Ajaw", "day": 9, "month": "Kumk'u"},
        {"number": 3, "name": "Kawak", "day": 7, "month": "Kumk'u"},
    ]}


def format_output(record, prediction):
    if len(prediction) != 2*len(record["dates"]):
        raise ValueError("Expected a year index and a day ordinal per designation")
    offsets = []
    for year, ordinal in zip(prediction[::2], prediction[1::2], strict=True):
        if year == ordinal == NO_DATE:
            offsets.append(None)
        elif 0 <= year < 52 and 0 <= ordinal < 365:
            offsets.append(365*year + ordinal)
        else:
            raise ValueError("Invalid year/ordinal prediction")
    return {"offsets": offsets}
