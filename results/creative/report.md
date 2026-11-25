# Paper folding, paragraph typesetting and film-club ballots

Three hand-programmed attention models produced **768/768 complete outputs** and **8,448/8,448 individual labels** on a Kaggle Tesla T4. A complete output requires every visible cell, line-break flag or elimination and winner to match. At sizes above the development range, 384/384 complete outputs matched.

## Held-out results

Development used sizes 4 and 8. The timed runs used FP32, one warm-up batch and 3 synchronized passes over each cohort of 64 records, in batches of 8. The table shows the median wall time for the entire cohort. Input preparation, host-to-device transfer, decoding and scoring were outside the timer. The stored parameter count includes zero entries and counts a shared tensor once. It excludes fixed input features, masks, indexing, and workspace. [All timings in seconds](sizes.csv)

| Application | Input size | Stored parameters | Exact records | Correct labels | Median ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Double-sided paper folding | 4×4 sheet | 26 | 64/64 | 256/256 | 15.12 |
| Double-sided paper folding | 8×8 sheet | 26 | 64/64 | 1,024/1,024 | 15.69 |
| Double-sided paper folding | 16×16 sheet | 26 | 64/64 | 1,024/1,024 | 29.44 |
| Double-sided paper folding | 32×32 sheet | 26 | 64/64 | 1,024/1,024 | 44.97 |
| Paragraph typesetting | 4 words | 66 | 64/64 | 256/256 | 14.63 |
| Paragraph typesetting | 8 words | 66 | 64/64 | 512/512 | 25.00 |
| Paragraph typesetting | 16 words | 66 | 64/64 | 1,024/1,024 | 46.71 |
| Paragraph typesetting | 32 words | 66 | 64/64 | 2,048/2,048 | 80.49 |
| Film-club ballot counting | 4 voters | 5 | 64/64 | 320/320 | 9.38 |
| Film-club ballot counting | 8 voters | 5 | 64/64 | 320/320 | 9.86 |
| Film-club ballot counting | 16 voters | 5 | 64/64 | 320/320 | 10.13 |
| Film-club ballot counting | 32 voters | 5 | 64/64 | 320/320 | 10.04 |

## When greedy wrapping or first choices give a different answer

In **78/256 paragraphs**, the reference layout had strictly lower squared raggedness than greedy wrapping. The model placed every line break correctly on **78/78** of those paragraphs. Equal-cost layouts are excluded from this subset.

Of 256 film-club elections, 206 had one film with more first-choice votes than every other film. In **85/256 elections**, a different film ultimately won. The model correctly predicted all four eliminations and the winner on **85/85** of those elections. These frequencies describe the generated test cases.

## Examples from the held-out run

### Fold a double-sided print

On this 4×4 sheet, fold the left half over the right half, then fold the bottom half under the top half. The two input faces share coordinates as viewed from the original front of the sheet. The rightmost grid shows the GPU’s predicted top surface.

| Original front | Original back | Visible after folding |
| --- | --- | --- |
| ⬛⬛⬜⬜<br>⬛🟨🟪🟨<br>🟦🟨🟪🟧<br>🟩🟥🟨🟦 | 🟥🟩⬜🟦<br>🟪🟧🟨🟥<br>🟨🟥🟧🟦<br>🟩🟦🟪🟩 | 🟩🟥<br>🟧🟪 |

The model returned color IDs `[[4, 1], [2, 6]]` for test record `folds_4[3]`. 0 = ⬛ 1 = 🟥 2 = 🟧 3 = 🟨 4 = 🟩 5 = 🟦 6 = 🟪 7 = ⬜.

Each repeated block uses the crease geometry to attend to two paper stacks. It turns the moving stack, then combines the exposed faces. The reference simulator preserves every individual paper layer. The study covers centered, axis-aligned folds of rectangular, eight-color sheets.

### Break a paragraph into lines

Test record `typeset_4[40]` has a 6-column text box. A greedy wrapper fills each line with as many words as fit. The transformer considers how each break affects the remaining lines.

| Method | Lines | Squared raggedness |
| --- | --- | ---: |
| GPU prediction | `fol`<br>`fl un`<br>`jrhcs` | 10 |
| Greedy wrapping | `fol fl`<br>`un`<br>`jrhcs` | 16 |

The cost is the sum of squared unused columns on all lines before the final line. Ties favor the longest possible earlier line. Each word uses one column per ASCII character, and adjacent words are separated by one space. The synthetic words exercise line lengths and break placement.

### Choose a film-club screening

In test record `ballots_8[0]`, eight members ranked five films. The GPU predicted **The Clockwork Orchard** as the winner. Recounting the ballots after each predicted elimination gives these vote transfers.

| Round | Moonlit Post Office | Paper Comet | Lantern Atlas | The Clockwork Orchard | Moth Orchestra | GPU elimination |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 2 | 3 | 2 | 0 | Moth Orchestra |
| 2 | 1 | 2 | 3 | 2 | — | Moonlit Post Office |
| 3 | — | 2 | 3 | 3 | — | Paper Comet |
| 4 | — | — | 3 | 5 | — | Lantern Atlas |

Each ballot supports its highest-ranked remaining film. The earliest nomination among the lowest-count films loses an elimination tie. The club’s rule continues through all four eliminations, including when a film already has a majority.

## Reproduce the results

Run these commands from the repository root. The prediction commands use NumPy and PyTorch on CPU. Replace `--example` with `--input FILE.json` to run your own record.

```sh
python creative/experiment.py predict folds --example
python creative/experiment.py predict typeset --example
python creative/experiment.py predict ballots --example
```

[Private Kaggle run](https://www.kaggle.com/code/ajayasaithambi/loopedwsp-creative-applications/versions/1) · [Run verification](<../../kaggle/runs/creative/v1/verification.json>) · [Frozen weights and data manifest](<../../creative/freeze.json>) · [Frozen cases](<../../creative/cases.json.gz>) · [Per-record predictions](<../../kaggle/runs/creative/v1/output/creative-predictions.npz>) · [Full GPU results](<../../kaggle/runs/creative/v1/output/creative-results.json>)

To recompute the expected answers and regrade all published predictions with NumPy, run the audit from the repository root. [Audit results](audit.json)

```sh
python creative/audit.py kaggle/runs/creative/v1 audit-output/creative-audit.json
```
