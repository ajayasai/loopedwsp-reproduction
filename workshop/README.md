# Kinetic mobiles, animation cels and punched music rolls

These three hand-programmed transformers solve an inverse balance problem, render a stack of transparent rectangles and decode damaged musical notation. Each accepts a JSON record. Two can also export the model output as an image or audio file.

The frozen models each produced 256/256 exact records on a Tesla T4. The report includes an actual counterweight calculation, a rendered image and four seconds of decoded music from the held-out run. [Results and examples](../results/workshop/report.md)

## Find the missing counterweight

Consider a mobile with four ornaments. From left to right, their masses are 2 grams, a missing weight, 3 grams and 5 grams. The three rods have left/right arm lengths of `[4,3]` at the root, `[2,1]` at the left joint and `[5,3]` at the right joint. The missing **4-gram** weight balances all three joints. At the root the two sides then exert proportional torques of 4 × 6 and 3 × 8.

The input describes a complete binary tree with 4, 8, 16 or 32 ornaments. Use `null` for the missing ornament, list the others left to right, and list the pairs of arm lengths by level starting at the root. The transformer evaluates the 16 candidate counterweights in parallel, carrying total mass and torque error up the tree with two attention heads. It returns `null` when no integer from 1 to 16 grams can balance every joint. The physical model assumes massless horizontal rods and ideal pivots. [Model and input bounds](mobile.py) · [Torque and equilibrium](https://openstax.org/books/college-physics-2e/pages/9-2-the-second-condition-for-equilibrium)

## Render an animation cel stack

Start with an opaque 8×8 background and add up to 32 colored rectangular cels, bottom to top. Coordinates are measured from the upper left and exclude the right and bottom bounds. Each RGB channel ranges from 0 to 15. Opacity `0`, `1` or `2` means clear, half or fully opaque. The compositor rounds halves upward after every layer. For example, placing a half-blue `[1,7,15]` stripe over opaque yellow `[15,13,4]` gives `[8,10,10]` at the overlap.

The model applies one shared attention block per cel and returns all 192 channel values. The PNG exporter maps the four-bit colors to eight-bit RGB and enlarges the frame with nearest-neighbor sampling. [Model and input format](cels.py)

## Recover a punched melody

Each string of eight observed holes represents one beat in a synthetic music-roll format. It can represent a rest or one of 15 notes from C4 through C6. The four event bits and four check bits form an extended Hamming code. Within a limit of two damaged holes per frame, the decoder returns the note and repaired track after a single error, and marks a double error unreadable. It needs only the observed holes. Three applications of the same attention block compute the parity checks before a readout selects the note and status. The reference instead compares the observed pattern with all 16 pristine codewords. [Model and bit layout](roll.py) · [Extended Hamming codes](https://www.ece.unb.ca/tervo/ee4253/hamming.shtml)

The WAV export uses one quarter-second slot per frame. It plays recovered notes and preserves the timing of rests and unreadable frames with silence. The JSON output distinguishes a rest from an unreadable frame and identifies each repaired track.

## Run your own input

Use Python 3.12 or later with NumPy and PyTorch installed. Run these commands from the repository root. The example and export commands use CPU.

```sh
python workshop/experiment.py predict mobile --example
python workshop/experiment.py predict cels --example
python workshop/experiment.py predict roll --example
python workshop/export.py cels frame.png --example
python workshop/export.py roll melody.wav --example
```

Replace `--example` with `--input FILE.json` to use your own input. For the counterweight example, save the following record and run `python workshop/experiment.py predict mobile --input mobile.json`.

```json
{"masses":[2,null,3,5],"arms":[[4,3],[2,1],[5,3]]}
```

## Frozen evaluation

Development uses eight records at each of sizes 4 and 8. Testing uses 64 records at each of sizes 4, 8, 16 and 32 per task. Size counts ornaments, cels or music frames. An exact record requires the whole answer, including every RGB channel or every note and repair status. The generator deduplicates encoded inputs across development and test within a task. Each timed measurement covers 64 GPU-resident records, in batches of eight, after a warm-up. The result includes all three synchronized forward-pass times and their median. Parameter totals count the actual stored scalar entries including zeros. Fixed representations, operators and workspace are outside those totals. [Protocol](protocol.json) · [Model ledgers](freeze.json) · [Frozen cases](cases.json.gz) · [Weights](weights.npz)

```sh
python workshop/experiment.py dev
python workshop/experiment.py gpu --output audit-output/workshop-gpu
python workshop/audit.py kaggle/runs/workshop/v1 audit-output/workshop-audit.json
```

To repeat the private Tesla T4 run on your own Kaggle account, authenticate the Kaggle CLI, set `LOOPEDWSP_KAGGLE_OWNER` to your username, and run `python workshop/submit_study.py --push`. Use `python workshop/submit_study.py --collect VERSION` to retrieve a completed version and check the source and output hashes. [Kaggle launcher](submit_study.py)
