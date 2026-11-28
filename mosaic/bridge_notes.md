# Choose an opening lead with two tricks left

With all four hands visible, which card should the leader play? This model counts the remaining tricks North–South can force and lists every optimal opening card for the player on lead. North and South maximize their partnership’s total. East and West minimize it. Knowing all four hands is called double-dummy play.

```sh
python mosaic/experiment.py predict bridge --input mosaic/examples/bridge.json
```

North holds the king of clubs and the six of spades. In no-trump, leading the king loses both tricks. East and South must follow clubs, West wins with the ace and cashes the jack of hearts. Leading the six of spades wins a trick immediately because nobody else has a spade. West can retain the ace of clubs to take the second trick. If hearts are trumps, East–West can take both tricks against either opening.

```json
{"solutions":[{"ns_tricks":1,"optimal_leads":["S6"]},{"ns_tricks":0,"optimal_leads":["CK","S6"]}]}
```

## Input and rules

Supply a list of 1–64 positions in `deals`. A position gives `hands` with two cards each for N, E, S and W, the `leader` and the `trump` suit or `NT` for no-trump. A card consists of its suit (C, D, H or S) and its rank (2–9, T, J, Q, K or A). All eight remaining cards must be distinct. Each position starts at the beginning of the penultimate trick. The model assumes ordinary legal play, no penalties, and perfect information.

Play proceeds clockwise. A player must follow suit if possible. Otherwise any card is legal. The highest trump wins a trick containing any trumps. Otherwise the highest card of the led suit wins. The winner leads the last trick. The returned trick count always belongs to North–South, even on an East–West lead. The optimal leads retain the card order in the leader’s input hand. [World Bridge Federation, Law 44](https://www.worldbridge.org/wp-content/uploads/2020/02/2017LawsofDuplicateBridge-gender_neutral_1.pdf)

## How the attention works

There are 16 ways for the four players to choose their first card from two. The transformer computes each branch’s two tricks in parallel. A softmax head prioritizes cards by trump, led suit and rank. It reads the winning partnership and the winner’s remaining suit to determine the lead for the second trick. Three masked attention reads then work backward from the last player’s decision through the other replies. Each reads the best legal continuation for the partnership acting at that point. A final comparison retains both opening cards when they tie.

The model stores three FP32 coefficients for ranking cards and sharpening the attention reads. Branch enumeration, card encoding and the follow-suit mask also contribute to the computation. The independent reference recursively plays every legal card. The benchmark mixes positions drawn from decks containing one, two or four suits and includes both trump and no-trump contracts. Increasing the number of positions increases the workload while each deal remains a two-trick ending. [Program and reference](bridge.py) · [Worked inputs and answers](checks.json)
