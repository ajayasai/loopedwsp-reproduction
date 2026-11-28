# Count the ways a tabletop arm can reach a point

An inverse kinematics problem asks which joint settings put a robot's tip at a desired position. This four-link arm moves on a square lattice. Its 81 possible joint settings are small enough to check exhaustively. The model returns the number of collision-free poses at each target and the first one in a fixed order. The experiment concerns static poses. Moving between poses would require a separate motion planner. [Lynch and Park on inverse kinematics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/inverse-kinematics-of-open-chains/)

```sh
python prism/experiment.py predict arm --input prism/examples/arm.json
```

In the example, all four links have length one. With a clear board, six poses reach `[2,2]`. They use every ordering of two steps east and two steps north. An obstacle at `[1,0]` rules out all three poses that begin by going east. The example returns three remaining poses and the first witness, `LSRS`. That arm goes north, north, east, east. The obstacle also blocks the only pose reaching `[4,0]`. Returning to the base at `[0,0]` would make the arm touch itself.

## Specify a board

The base is at `[0,0]` and initially faces east. Positive x points east and positive y points north. Each of the four joints turns left 90 degrees (`L`), stays straight (`S`), or turns right 90 degrees (`R`) before its link extends. Each link in `lengths` must be one or two units long. A pose touches every lattice point along its links, including the base, tip and midpoint of a length-two link. Reusing a point after leaving it or touching a blocked point invalidates that pose.

Supply up to eight distinct `obstacles` and 1–64 `targets` as integer coordinate pairs in the range −8 to 8 on each axis. A target constrains tip position only. For each target, `count` is the number of feasible joint words, and `first_pose` gives the first under the order L, S, R. It is `null` when the count is zero. Targets are independent requests on the same board.

## How it works

The transformer enumerates the 81 candidate words and uses attention to read direction vectors for their cumulative quarter turns. The supplied lengths determine the points occupied by every link. At each point an attention head reads the squared distance to the nearest obstacle. Repeated-point checks remove self-intersections. A final head selects the earliest feasible word at each target, and summing its target's feasible-candidate mask gives the count. The model stores nine FP32 coefficients, including eight entries in its direction table. Candidate enumeration and collision checks also use fixed operations and working memory.

The independent reference grows paths in joint order and prunes a branch as soon as it repeats a point. The benchmark varies all four link lengths, the obstacle locations and the requested points. Targets are sampled across unique solutions, multiple solutions, obstacle-blocked locations and locations the arm cannot reach even on an empty board, where the board supplies those categories. [Model and generator](arm.py) · [Worked checks](checks.json)
