# Trace a contour through an elevation grid

A contour enters and leaves a map cell through edges whose endpoints lie on opposite sides of the requested elevation. Two such edges give one connection. Four crossed edges can give two separate branches or one four-way junction. This model resolves each cell under bilinear interpolation of its four samples. The asymptotic decider in the original paper handles the same ambiguity on the faces of a sampled 3D field. [Nielson and Hamann, *The Asymptotic Decider*](https://graphics.stanford.edu/courses/cs164-10-spring/Handouts/paper_p83-nielson.pdf)

```sh
python frontier/experiment.py predict contours --input frontier/examples/contours.json
```

The example traces elevation 100 through two rows of samples, `[104,99,101,98]` and `[99,101,99,101]`. All three cells have alternating corner signs relative to 100. The first pairs north with east and south with west. The middle contains a four-way junction. The last pairs west with north and east with south. Its JSON result is

```json
{"cells":[["N-E & S-W","four-way saddle","W-N & E-S"]],"four_way_saddles":[[0,1]]}
```

Supply an integer `level` from −4096 through 4096 and a rectangular `heights` array with 2–65 vertex samples on each axis. Each integer sample must lie between one and eight elevation units above or below the level. A vertex exactly on the contour would require its own degeneracy convention. Rows run north to south and columns west to east. Junction coordinates give the zero-based cell row and column. The output records local connectivity under the assumed interpolation. It does not estimate terrain between surveyed points independently of that assumption.

Two position-addressed attention reads gather the four elevations for each cell. The sign pattern selects a programmed case. In a four-crossing cell, the signs alone leave the connectivity open. For clockwise level-relative corners *a, b, c, d* starting at northwest, the model evaluates *ac − bd* with a ReLU network that squares integers exactly. Its sign selects a pairing, and zero gives the junction. The separate reference finds the exact fractional crossing positions on the north and south edges. The benchmark combines slopes, hills, irregular fields and grids with deliberately planted saddles. [Model](contours.py) · [Worked cases](checks.json)
