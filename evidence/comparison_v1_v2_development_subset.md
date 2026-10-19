# v1 vs v2 on development

Identical inputs and scoring: file_hashes=yes, settings=yes, model=yes, provider=yes, evaluator_version=yes
Prompt versions: v1-dcdb8b4b15 -> v2-73399f0449

| metric | baseline | candidate | delta |
|---|---|---|---|
| extraction precision | 1.0000 | 1.0000 | +0.0000 |
| extraction recall | 0.8850 | 0.9735 | +0.0885 |
| extraction f1 | 0.9390 | 0.9865 | +0.0476 |
| joint precision | 1.0000 | 0.9909 | -0.0091 |
| joint recall | 0.8850 | 0.9646 | +0.0796 |
| joint f1 | 0.9390 | 0.9776 | +0.0386 |
| gold claim coverage | 0.8850 | 0.9646 | +0.0796 |
| macro F1 | 0.9372 | 0.9765 | +0.0393 |
| supported precision | 1.0000 | 0.9737 | -0.0263 |
| supported recall | 1.0000 | 1.0000 | +0.0000 |
| supported f1 | 1.0000 | 0.9867 | -0.0133 |
| contradicted precision | 1.0000 | 1.0000 | +0.0000 |
| contradicted recall | 1.0000 | 0.9143 | -0.0857 |
| contradicted f1 | 1.0000 | 0.9552 | -0.0448 |
| unknown precision | 1.0000 | 1.0000 | +0.0000 |
| unknown recall | 0.6829 | 0.9756 | +0.2927 |
| unknown f1 | 0.8116 | 0.9877 | +0.1761 |

| count | baseline | candidate |
|---|---|---|
| P | 100 | 110 |
| M | 100 | 110 |
| C | 100 | 109 |
| unmatched_predictions | 0 | 0 |
| answer_errors | 0 | 0 |
| error types | {} | {} |

## Confusion matrices (rows expected, columns predicted)

baseline:

| expected | supported | contradicted | unknown | missing_or_error |
|---|---|---|---|---|
| supported | 37 | 0 | 0 | 0 |
| contradicted | 0 | 35 | 0 | 0 |
| unknown | 0 | 0 | 28 | 13 |
| no_gold | 0 | 0 | 0 | 0 |

candidate:

| expected | supported | contradicted | unknown | missing_or_error |
|---|---|---|---|---|
| supported | 37 | 0 | 0 | 0 |
| contradicted | 1 | 32 | 0 | 2 |
| unknown | 0 | 0 | 40 | 1 |
| no_gold | 0 | 0 | 0 | 0 |

## Claim-level changes: 13 fixed, 4 regressed, 0 still wrong but changed

Fixed:
- A03:335:505: missed -> correct
- A03:506:687: missed -> correct
- A04:1984:2165: missed -> correct
- A04:2166:2361: missed -> correct
- A04:494:694: missed -> correct
- A04:696:899: missed -> correct
- A08:1084:1282: missed -> correct
- A08:1479:1676: missed -> correct
- A08:2201:2395: missed -> correct
- A08:687:889: missed -> correct
- A12:1379:1559: missed -> correct
- A12:335:515: missed -> correct
- A12:676:871: missed -> correct

Regressed:
- A04:2363:2540: correct -> missed
- A09:492:655: correct -> wrong:supported
- A10:1462:1643: correct -> missed
- A10:1644:1821: correct -> missed

Changed but still wrong:
- none

Extra predictions: baseline 0, candidate 0; new in candidate []; removed []

## Per-answer joint F1 changes

| answer | baseline | candidate | delta | status |
|---|---|---|---|---|
| A03 | 0.875 | 1.000 | +0.125 | ok -> ok |
| A04 | 0.778 | 0.952 | +0.175 | ok -> ok |
| A08 | 0.750 | 1.000 | +0.250 | ok -> ok |
| A09 | 1.000 | 0.889 | -0.111 | ok -> ok |
| A10 | 1.000 | 0.889 | -0.111 | ok -> ok |
| A12 | 0.800 | 1.000 | +0.200 | ok -> ok |
