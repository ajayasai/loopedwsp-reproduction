# Reading an uncontracted English Braille label

The same raised dots can stand for a letter or a digit. For example, dots 1 and 2 mean **b** in ordinary text, **B** following a single capital indicator, and **2** in a number. This reader covers a bounded subset of uncontracted Unified English Braille (UEB). Its inputs are the raised dots of each cell in reading order. A physical display, image recognition or a transcription service would supply those dots. [International Council on English Braille rulebook, sections 5, 6 and 8](https://iceb.org/wp-content/uploads/2025/10/Rules-of-Unified-English-Braille-2024.pdf)

```sh
python mosaic/experiment.py predict braille --input mosaic/examples/braille.json
```

The example prints `{"text":"Room 204b"}`. Dot 6 capitalizes the first **r**. Dots 3456 introduce the number, so the patterns for **b**, **j** and **d** read **2**, **0** and **4**. Dots 56 then introduce a grade-1 letter and the next **b** reads as a letter again.

## Input and output

Supply 1–128 cells. Each string contains the raised dot numbers in ascending order. Use `""` for a blank. Viewed from the reading side, dots 1, 2 and 3 run down the left column, and 4, 5 and 6 run down the right.

```json
{"cells":["6","1235","135","135","134","","3456","12","245","145","56","12"]}
```

Inputs are correctly transcribed lines using the 26 English letters, single-capital indicators, unspaced integer digit sequences, the grade-1 symbol indicator before a lowercase a–j after a number, and ordinary blanks. Within this repertoire, a blank, an ordinary k–z cell, a capital indicator or a grade-1 indicator ends numeric mode. Single-capital indicators apply to the next letter. Contractions, punctuation, decimals, word or passage indicators, formatting and line wrapping require further rules. The CLI rejects unknown dot patterns and an indicator lacking an appropriate following cell. It assumes the source supplied the intended UEB indicators.

The model returns a prediction for each input cell. Class 0 consumes an indicator without printing a character. Class 1 is a printed blank. Classes 2–27 give lowercase a–z, 28–53 give uppercase A–Z, and 54–63 give digits 0–9. The formatter joins the visible symbols to produce the text. Record accuracy requires every aligned class, including the indicators, to match.

## How the weights work

The encoder passes six occupancy bits per cell. The first attention lookup matches each six-dot pattern to a glyph with ordinary, uppercase and numeric readings. A causal attention head finds the latest earlier cell that can start or end numeric mode. This allows a distant number sign to govern a later digit. The previous cell supplies the single-capital flag. The network stores 127 FP32 coefficients for bit weights, glyph keys, reading values and attention sharpness. Positions and the causal mask are fixed.

The independent reference reads the cells from left to right with explicit numeric and capital state. Benchmark lines include long numbers, numbers followed by grade-1 letters, k–z letters that terminate numeric mode, and spaces between text and numbers. They are generated rule tests. [Model](braille.py) · [Worked checks](checks.json)
