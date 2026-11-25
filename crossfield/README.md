# Turkish morphology, crystal diffraction and auction design

These three hand-programmed transformers take JSON inputs. One chooses the endings on a Turkish noun. One predicts the strength of a reflection from an idealized crystal. One allocates goods to bidders and computes each winner’s price. Each matched all 256 held-out records on a Kaggle Tesla T4. [Measured results by size](../results/crossfield/report.md) · [Independent regrading](../results/crossfield/audit.json)

## Run the examples

From the repository root, use Python 3.12 or later with NumPy and PyTorch installed. These commands run on CPU. Substitute your own JSON file, or omit `--input` to read from standard input.

```sh
python crossfield/experiment.py predict inflection --input crossfield/examples/inflection.json
python crossfield/experiment.py predict diffraction --input crossfield/examples/diffraction.json
python crossfield/experiment.py predict auction --input crossfield/examples/auction.json
```

| Field | What to look for in the example |
| --- | --- |
| [Turkish inflection](examples/inflection.json) | The accusative of *göl* (“lake”) is *gölü*. Its plural accusative is *gölleri*. The plural ending supplies a new vowel for the following suffix. |
| [Diffraction](examples/diffraction.json) | Four equal sites in a face-centered cubic arrangement give intensities **0, 0, 16, 16** for the (100), (110), (111) and (200) reflections. |
| [Bundle auction](examples/auction.json) | Bids of 10 for A and 9 for B together beat a bid of 17 for AB. The separate winners pay **8** and **7**. An uncontested bidder for C wins and pays **0**. |

## Computational linguistics

Each request contains a lowercase `stem`, `number` (`singular` or `plural`) and `case` (`nominative`, `locative`, `ablative`, `accusative` or `genitive`). Supply 1–32 requests in `requests`, with 1–24 Turkish letters and at least one vowel per stem.

Attention reads the last letter and the last vowel to select the number ending, then reads the extended stem to select the case ending. This handles front/back and rounding harmony, the buffer consonants *y* and *n*, and the choice of *d* or *t*. The model predicts two suffix IDs per request. The formatter appends their spellings to the supplied stem. Its rule applies to regular common nouns whose stem stays unchanged. Stem-final alternation, vowel loss, exceptional harmony, possessives and proper names require additional rules or lexical information. The generated benchmark uses synthetic stems. The separate worked check uses real words. [Implementation](inflection.py) · [Turkish inflectional suffixes](https://web.itu.edu.tr/adali/wp-content/uploads/2021/10/T%C3%BCrk%C3%A7enin-%C3%87ekim-Ekleri.pdf)

## Materials science

Define 1–8 distinct scattering `sites` as `[x,y,z,amplitude]`. Coordinates are integers 0–3 in **quarter-cell units**. Amplitudes are 1 or 2. Supply 1–32 signed Miller-index triples in `reflections`, with each component from −4 to 4. For each triple, the model returns the squared magnitude of the sum of the sites’ complex scattering amplitudes, preserving the input order.

The first attention operation selects a phase for each site and reflection. The second combines the real and imaginary contributions. A programmed readout computes their squared magnitude. The model assumes constant real scattering amplitudes, ideal discrete point sites and kinematical diffraction. A zero can be specific to the motif and its contrast. For example, changing the body-center amplitude from 1 to 2 changes the model (100) intensity from 0 to 1. Assigning a space group or predicting a measured powder pattern would require additional information. [Implementation](diffraction.py) · [IUCr definition of the structure factor](https://dictionary.iucr.org/Structure_factor)

## Market design

There is one unit each of goods A, B, C and D. Supply 1–8 bids in `bids`. Each bidder wants one nonempty `bundle` written in alphabetical order, with an integer `value` from 1 to 20. Winners receive disjoint bundles that maximize total reported value. Ties favor the smallest integer winner mask, using the first bidder as the least-significant bit. The result gives a separate win flag and payment for each input bidder, plus total reported value as `welfare`.

For winner *i*, the price is the other bidders’ best total value if *i* were absent, minus their value in the selected allocation. Losers pay zero. Attention selects the best feasible subset for the full auction and for every leave-one-bidder-out auction. The model stores six coefficients and enumerates every candidate subset, reaching 256 subsets for eight bidders. [Implementation](auction.py) · [Clarke pivot pricing](https://www.cs.cornell.edu/~rafael/networks-html/chapter10.html)

## Evaluation

| Model | Stored coefficients | Development sizes | Held-out sizes | Unit of size |
| --- | ---: | --- | --- | --- |
| Inflection | 449 | 2, 4 | 2, 4, 8, 16 | Noun requests per record |
| Diffraction | 43 | 4, 8 | 4, 8, 16, 32 | Reflections per motif |
| Auction | 6 | 2, 4 | 2, 4, 6, 8 | Bidders per auction |

There are eight development records and 64 held-out records at each task and size. Each record must match in its entirety. Identical encoded records are excluded across splits within each task. Longer inflection lists and reflection lists repeat the operation over more queries; bidder count also changes the size of the auction’s candidate space. Counts include stored coefficient entries and exclude fixed features, masks, enumerated candidates and working memory. [Protocol](protocol.json) · [Frozen cases](cases.json.gz) · [Model and data fingerprints](freeze.json) · [Worked checks](checks.json)

```sh
python crossfield/experiment.py dev
python crossfield/experiment.py gpu --output audit-output/crossfield-gpu
python crossfield/audit.py kaggle/runs/crossfield/v1 audit-output/crossfield-audit.json
```

To run on another Kaggle account, authenticate the Kaggle CLI, set `LOOPEDWSP_KAGGLE_OWNER` to your username, and run `python crossfield/submit_study.py --push`. Collect its completed version with `python crossfield/submit_study.py --collect VERSION`. The launcher requests a private Tesla T4 with internet disabled and checks the server-side script and output hashes. [Kaggle launcher](submit_study.py)
