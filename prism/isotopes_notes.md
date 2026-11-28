# Predicting overlapping isotope peaks

Give the model a mixture of singly charged ions with known light-isotope masses and halogen counts. It predicts the ideal intensity in each nominal mass-to-charge bin from 400 through 415. Two ions with different light-isotope masses can contribute to the same peak.

```sh
python prism/experiment.py predict isotopes --input prism/examples/isotopes.json
```

In the example, a component with two chlorine atoms at 400 gives intensities 36, 24 and 4 at bins 400, 402 and 404. A bromine component at twice the abundance contributes another 64 to each of bins 402 and 404. A third component contributes 64 at 401. The model therefore returns

```json
{"mz_min":400,"intensities":[36,64,88,0,68,0,0,0,0,0,0,0,0,0,0,0]}
```

## Input and units

The `ions` list contains 1–16 components. Each component specifies four integers. The `mz` field gives the nominal mass-to-charge ratio with all halogens in their light isotope, from 400 to 409. The `cl` and `br` fields give chlorine and bromine atom counts, with up to three halogens altogether. Relative `abundance` ranges from 1 to 3. Each unit contributes 64 intensity units summed over all mass bins. Repeated components contribute additively.

The synthetic spectrum uses independent isotope choices, with heavy-isotope probabilities of one quarter for chlorine and one half for bromine. Replacing a light halogen with its heavy isotope adds two nominal mass units. These rounded teaching ratios approximate natural abundances. The rest of each ion contributes one fixed mass. Exact isotope masses, other isotope satellites, charge states, instrument response, fragmentation, and measurement noise would require a more detailed model. [CIAAW chlorine isotope abundances](https://ciaaw.org/chlorine.htm) · [CIAAW bromine isotope abundances](https://ciaaw.org/bromine.htm)

## How the weights work

The encoder carries the four input values per component. The network puts its initial intensity in the all-light bin. At each of three shared attention steps, the current atom selects the weights for two source bins. Intensity can stay at its present mass or arrive from two units below. The relative attention weights are 3 to 1 for chlorine and 1 to 1 for bromine. A padding step keeps the existing distribution. Summing the resulting component spectra gives the mixture. Three frozen FP32 scalars store the heavy-branch log odds. Source offsets, bin addresses, loop depth, and the nearest-integer readout are fixed operations.

An independent reference counts isotope substitutions using binomial coefficients. The worked examples include a chlorine doublet, a bromine doublet, an overlapping mixture and a mixed-halogen peak at the upper bin boundary. [Model](isotopes.py) · [Worked inputs and targets](checks.json)
