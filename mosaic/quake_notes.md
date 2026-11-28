# Locate a synthetic seismic source from first arrivals

Given the first arrival time at several stations, the model finds every grid position and origin time that fits the observations. This experiment uses a flat 7×7 grid and a constant wave speed of one grid unit per tick. At a station a Euclidean distance *d* from the source, the first recorded tick is the integer origin tick plus ⌈*d*⌉. The origin can be any integer from 0 to 15. Real earthquake location uses a crustal velocity model and estimates depth as well as horizontal position and origin time. [USGS overview of earthquake location](https://www.usgs.gov/faqs/how-do-seismologists-locate-earthquake)

```sh
python mosaic/experiment.py predict quake --input mosaic/examples/quake.json
```

The three stations in the example are at (0, 0), (6, 0), and (0, 6), with arrivals at ticks 8, 9, and 8. From (2, 3), their travel times round up to 4, 5, and 4 ticks. Subtracting these delays from the arrivals gives the same origin tick of 4 at all three stations. The example returns

```json
{"solutions":[{"x":2,"y":3,"origin_tick":4}]}
```

If the third arrival is missing, the other two readings allow five solutions. One of them starts at (2, 4) at tick 3. A pair of stations one grid unit apart reporting ticks 8 and 10 admits no solution under the model.

## Input and results

Supply `stations` as 1–49 triples `[x,y,first_arrival_tick]` with distinct station coordinates. Both coordinates must be integers from 0 to 6. A measured tick is an integer from 0 to 24. Supply `null` for a missing measurement and include at least one measured station. The model lists every solution, in order of increasing *y*, then *x*. An empty list means no origin and position explain all the supplied picks. The result assumes exact station coordinates and the declared sampling rule. The simulation does not model depth, refraction or uncertainty in the readings.

## Program and evaluation

For each candidate source and station, the squared distance addresses an attention memory storing travel times for all integer squared distances from 0 to 72. Two attention reads then take the earliest and latest origin ticks implied by the measured stations. A source cell qualifies when those times agree within the half-tick integer decision boundary and fall in the allowed range. The 74 fixed FP32 coefficients comprise 73 travel-time entries and one attention scale. The model also uses fixed coordinate arithmetic, a missing-data mask and a class readout.

The model emits one class per grid position. Class 0 excludes the position and classes 1–16 encode origin ticks 0–15. The record score requires all 49 position and time classes to agree with the reference. The generator mixes consistent networks, missing picks, layouts chosen to preserve ambiguity, one-tick perturbations and incompatible readings. Increasing the station count gives more constraints on the same source grid. [Model and generator](quake.py) · [Worked cases](checks.json)
