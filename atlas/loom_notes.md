# Weave a drawdown from a loom draft

A drawdown shows which yarn is on top at each crossing. Give the model a rising-shed draft and it returns the face of the cloth, row by row. Each `#` is a warp end passing over the weft. Each `.` is the weft passing over a warp end. [University of Colorado Boulder Unstable Design Lab’s guide to floor-loom drafts](https://unstable.design/books/prototyping-experimental-textiles/docs/resources/how-to-weave/)

From the repository root, run the four-shaft twill example.

```sh
python atlas/experiment.py predict loom --input atlas/examples/loom.json
```

The first pick raises shafts 1 and 2. The following picks raise 2 and 3, then 3 and 4, then 4 and 1. Repeating the sequence gives the diagonal lines of a two-over, two-under twill.

```text
##..##..
.##..##.
..##..##
#..##..#
##..##..
.##..##.
..##..##
#..##..#
```

The first warp end appears at the left, and the first pick appears at the top. The formatter also reports the longest run of exposed warp along one column and exposed weft along one row. Both lengths are 2 for this example. The counts stop at the boundary of the supplied rectangle.

## Write your own draft

- `threading` gives one shaft number (1–8) for each of 1–64 warp ends, from left to right.
- `tie_up` gives a list of lifted shafts for each of 1–8 treadles. The first list describes treadle 1. An empty list gives an unconnected treadle.
- `treadling` gives the treadles pressed together for each of 1–64 picks. Pressing two treadles lifts every shaft connected to either. An empty pick leaves all shafts down.

Numbers within a tie-up row or treadling row must be distinct. Rectangular swatches are supported. This model predicts the interlacement of an ideal rising-shed loom. Yarn thickness, color, tension and physical stability require additional information. Fully raised or lowered rows let the checks exercise extreme cases.

## How the attention works

The encoder stores the three draft charts in separate parts of an integer array. The first attention head uses an end’s shaft number to read that shaft’s connections to all treadles. For each crossing, a second attention head reads the active treadles and prefers any one that lifts the shaft. An always-available neutral treadle handles idle picks. The network stores four FP32 scalars for addressing and classification. The two longest floats are calculated directly from the decoded cloth.

The independent reference opens each shed by taking the union of shafts connected to pressed treadles, then checks every warp end for membership. The benchmark includes straight and point threadings, compound picks and random tie-ups. An exact draft requires every cell to match.
