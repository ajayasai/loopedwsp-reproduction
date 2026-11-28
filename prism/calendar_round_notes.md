# Resolving a Maya Calendar Round designation

The Tzolk’in pairs a cycle of 13 numbers with 20 day names. The Haab’ has eighteen 20-day months followed by the five days of Wayeb’. Together the 260- and 365-day cycles repeat every 18,980 days. This model takes a complete designation in both calendars and returns its zero-based offset in one round beginning at **4 Ajaw 8 Kumk’u**. [Smithsonian National Museum of the American Indian, *Living Maya Time*](https://maya.nmai.si.edu/calendar/calendar-system)

```sh
python prism/experiment.py predict calendar_round --input prism/examples/calendar_round.json
```

| Input designation | Offset | Reason |
| --- | ---: | --- |
| 4 Ajaw 8 Kumk’u | 0 | Both cycles match the origin. |
| 4 Ajaw 9 Kumk’u | `null` | Every recurrence of 4 Ajaw is 260 days apart. None can advance the Haab’ by exactly one day modulo 365, since both cycle lengths are divisible by five. |
| 3 Kawak 7 Kumk’u | 18,979 | The next day begins the next round. |

The example prints `{"offsets":[0,null,18979]}`. These are relative offsets. Assigning an absolute date to a recurring designation requires historical evidence beyond the two calendar labels.

## Input and spelling

Give a `dates` list of 1–64 objects. Each has four fields, as in `{"number":4,"name":"Ajaw","day":8,"month":"Kumk'u"}`. Use numbers 1–13 and Haab’ ordinals 0–19, or 0–4 in Wayeb’. JSON inputs use straight ASCII apostrophes. Spellings follow John Montgomery’s dictionary, one of several naming conventions. [Day names](https://research.famsi.org/montgomery_dictionary/daysigns.php) · [Month names](https://research.famsi.org/montgomery_dictionary/monthsigns.php)

The day names in order are `Imix, Ik', Akb'al, K'an, Chicchan, Kimi, Manik', Lamat, Muluk, Ok, Chuwen, Eb', B'en, Ix, Men, Kib', Kab'an, Etz'nab', Kawak, Ajaw`. The month names are `Pop, Wo, Sip, Sotz', Sek, Xul, Yaxk'in, Mol, Ch'en, Yax, Sak, Kej, Mak, K'ank'in, Muwan, Pax, K'ayab', Kumk'u, Wayeb'`.

## Program and evaluation

One attention head locates the requested number and name in a 260-position Tzolk’in memory. A linear projection and wrap operation convert the Haab’ date to its first ordinal on or after the epoch. Every later occurrence of that Haab’ date is another 365 days away. A second head looks up the number of 365-day intervals for which the two cycle phases agree. Its fallback token marks impossible pairs. The network emits the interval number and ordinal separately so that the output vocabulary needs only 366 classes, including the incompatibility sentinel. Adding the ordinal to 365 times the interval number recovers the day offset.

The independent reference tries the 52 possible occurrences of the requested Haab’ date within the round. The synthetic benchmark deliberately pairs compatible designations with near misses in which moving the day name by one to four positions breaks compatibility. It shuffles their order, and reserves some requests for Wayeb’. Accuracy at larger input sizes measures the ability to solve longer lists of designations with the same weights. [Worked cases](checks.json) · [Model and reference](calendar_round.py)
