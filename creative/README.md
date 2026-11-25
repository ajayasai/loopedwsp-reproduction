# Three creative applications of programmed transformer weights

Each application has a different input and produces a complete artifact: a folded color grid, a paragraph layout or the result of a film-club vote. All three use hand-set coefficients and reuse an attention block across steps. The test cases are synthetic. The model's stored weights, its fixed operations and the size of its working state all contribute to what it can compute.

## Paper-fold preview

Supply front and back color grids in the coordinates of the original front-up sheet. A crease specifies which half moves and whether it goes over or under the other half, such as `left_under_right`. The output is the visible grid after all folds. The model addresses the stationary cell and its reflected partner with two softmax attention heads, then selects the new outer faces. It keeps just the top and bottom colors of each stack. The reference simulator stores every layer and physically reverses the moving stacks. Each side may be 1, 2, 4, 8, 16 or 32 cells and each cell uses one of eight colors. [Model and input rules](folds.py)

## Paragraph typesetter

Supply a width and a list of words. The model chooses line breaks to minimize the sum of squared unused columns, with a free last line. Ties favor the longer first line, then the longer second line, and so on. At width 6, greedy wrapping gives `Ink on / my / plate` and costs 16. The better layout `Ink / on my / plate` costs 10. The tied block considers all word boundaries that fit, attending to the next boundary using the cost of the current line and the estimated cost of its suffix. The reference uses integer dynamic programming and reconstructs the best layout. Inputs allow up to 32 monospaced printable ASCII words and widths up to 32 columns, with one space between words on a line. [Model and input rules](typeset.py)

## Film-club ranked ballots

Supply five film titles in nomination order and each voter's complete ranking. Every round gives each ballot to its highest-ranked remaining film and removes the film receiving the fewest votes. The earliest nominee among tied films is eliminated. The model reports the full order of four eliminations and the winner. Three attention operations select each voter's current favorite, pool the votes and select an elimination. The reference counts integer votes over title strings. This particular club rule accepts 1–128 complete ballots and continues to the last film even when a film has already secured a majority. [Model and input rules](ballots.py)

## Try the models

From the repository root, use Python 3.12 or later with NumPy and PyTorch installed. These commands run on CPU and print both an example input and its model output.

```sh
python creative/experiment.py predict folds --example
python creative/experiment.py predict typeset --example
python creative/experiment.py predict ballots --example
```

To try your own paragraph, save the following as `paragraph.json` and run `python creative/experiment.py predict typeset --input paragraph.json`. Any model also accepts a JSON object on standard input.

```json
{"width": 6, "words": ["Ink", "on", "my", "plate"]}
```

## Frozen evaluation

| Application | What size counts | Development sizes | Held-out sizes | When a record counts as correct |
| --- | --- | --- | --- | --- |
| Paper folding | Cells per side of the original square | 4, 8 | 4, 8, 16, 32 | Every visible output cell has the correct color. The four sizes use 2, 2, 4 and 6 folds respectively. |
| Typesetting | Words in the paragraph | 4, 8 | 4, 8, 16, 32 | Every line break matches the optimal layout and its tie rule. |
| Ranked ballots | Voters, with five films throughout | 4, 8 | 4, 8, 16, 32 | All four eliminated films and the winner match in order. |

There are eight development records and 64 held-out records per task and size. The generator uses separate random streams and excludes repeated encoded inputs within a task, including across splits. The test set contains 768 records. Exact-record accuracy is the primary measure. Individual output labels, nonfinite outputs and wall time are recorded separately. Each timing covers 64 records already encoded on the device, in batches of eight, after a warm-up. The GPU run saves three synchronized forward-pass timings per setting. Tensor counts include stored zero entries and count shared weights once. [Inputs and oracle answers](cases.json.gz) · [Frozen tensors](weights.npz) · [Parameter and data hashes](freeze.json) · [Evaluator](experiment.py)

```sh
python creative/experiment.py dev
python creative/experiment.py gpu --output audit-output/creative-gpu
```

To submit to Kaggle, authenticate the Kaggle CLI, set `LOOPEDWSP_KAGGLE_OWNER` to your account, and run `python creative/submit_study.py --push`. Collect a completed run with `python creative/submit_study.py --collect VERSION`. The launcher requests a private Tesla T4 with internet access disabled and verifies the returned source and output hashes. [Kaggle launcher](submit_study.py)
