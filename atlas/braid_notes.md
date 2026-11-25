# Counting the colorings of a closed braid

Enter a word of up to 64 crossings on three strands, read from top to bottom. The generator `1` passes the left strand over the middle strand and `2` passes the middle over the right. A minus sign reverses that crossing. Join each bottom position to the same top position to close the braid. The empty word closes to three separate circles.

The runnable input `{"word":[1,1,1,2]}` represents a trefoil. It produces `{"fox_3_colorings":9,"nonconstant_colorings":6}`. At every crossing, twice the color on the overstrand must equal the sum of the incoming and outgoing understrand colors modulo three. The three constant assignments satisfy this rule for every input. Different links can have the same count. For example, the figure-eight knot and the unknot both have just the three constant colorings. [Fox coloring rule](https://doi.org/10.1007/s10623-025-01615-x)

The model starts with all 27 possible assignments of colors to the top positions. Each crossing token attends to the signed generator in a four-entry memory to select its strand transformation. A shared feedforward layer reduces the transformed colors modulo three. After the last crossing, the model counts the assignments that end with their original colors. It stores 50 FP32 coefficients. The three-strand enumeration and the rule that joins corresponding endpoints are fixed. The reference instead solves three homogeneous closure equations by row reduction over the field of three elements.

The study mixes randomly sampled words with constructed controls. Inserting adjacent inverse crossings leaves a braid unchanged. Cyclically shifting a word preserves its closure. Two separate four-crossing checks permute the same signed generators. The words `[1,1,2,2]` and `[1,2,1,2]` have 3 and 9 colorings respectively.

Run the trefoil example from the repository root.

```sh
python atlas/experiment.py predict braid --input atlas/examples/braid.json
```
