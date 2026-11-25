# Tree-ring pattern matching

Can an eight-year sample of ring widths be lined up with a longer reference series even when the two trees grow at different rates? This model slides the sample along the reference, compares years when both widths rose or both fell, and returns every start tied for the best agreement. In dendrochronology, researchers compare ring patterns across many specimens to establish calendar dates. This synthetic benchmark evaluates a relative alignment rule on one sample and one reference. [University of Arizona Laboratory of Tree-Ring Research](https://www.ltrr.arizona.edu/dendrochronology.html)

## Try an alignment

Run the example from the repository root.

```sh
python atlas/experiment.py predict rings --input atlas/examples/rings.json
```

The reference is `[60,70,65,50,58,55,62,66,61,63,50,43]`. The sample is `[133,103,119,113,127,135,125,129]`. Its first two rings get narrower and then wider, from 133 to 103 to 119. Reference positions 2–4 have the same changes, from 65 to 50 to 58. All seven changes match at start 2, where doubling each reference width and adding three gives the entire sample. The model returns `{"status":"unique","best_offsets":[2],"insufficient_offsets":[]}`. Offsets start at zero.

## Score and input limits

Supply eight entries in `sample` and 8–64 entries in `reference`, ordered oldest to newest. Each width is an integer from 1 to 255 in consistent arbitrary units within its series. Use `null` for an unmeasured year whose position is known. The alignment preserves that position.

At each offset, use a one-year interval only when all four endpoint widths are present and each series changes. Count the fraction of the usable intervals in which the directions agree. At least four of the seven intervals must qualify. Each eligible start that attains the highest fraction is returned. If several starts tie they are all listed, even if they used different numbers of intervals. If none qualifies, the status is `insufficient`. Completely unrelated series can yield a unique best score. One published control wins at just one agreement in seven. Dating actual material calls for independent chronology and corroborating evidence. The model assumes that annual positions are already known, including the locations of unmeasured widths.

## How the weights work

The input encoder emits the literal widths and uses zero for an unmeasured year. Position queries read the eight reference widths for every candidate offset through dot-product attention. A pointwise clipped difference encodes a rise, fall or excluded interval. Fixed sums count eligible and agreeing changes. A table of four exact multipliers converts the eligible fractions to integer multiples of 1/420. A second attention head reads the highest score across offsets and the readout marks every tie. The model stores eleven FP32 coefficients. Fixed positional features, comparisons and candidate memory also contribute to the computation. The independent evaluator compares rational numbers directly. [Model and reference](rings.py) · [Worked inputs and offset labels](checks.json)
