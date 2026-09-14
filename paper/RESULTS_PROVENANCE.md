# ICASSP results provenance and submission gate

## Numbers retained in the manuscript

The values in icassp2027.tex are the point estimates already present in the
author's draft. No value was changed to improve appearance. The manuscript now
labels the fixed split and discusses the two unfavorable regimes:

- Norman top-20 DE: MSE 0.291 versus 0.246; Pearson 0.856 versus 0.868.
- Adamson: Pearson 0.9894 versus 0.9897; DE MSE 0.193 versus 0.184.

## Claims removed because evidence was not traceable

- CKA values before and after fusion.
- Claims of significant support-mask and DEG overlap.
- The epistatic scatter and highlighted interaction pairs.
- The claim that 65 of 74 combinations improve over scPert.
- BioLLM as a third fusion source.
- Whole-model linear complexity and the unproved 6.02 dB bound.

Restore a removed figure only from saved per-sample predictions and an analysis
script that fails closed when predictions are unavailable.

## Required experiment gate before submission

Run the locked configuration for seeds 1, 2, 3, 4, and 5. Report mean plus
standard deviation for all-gene and top-20 DE metrics, plus a paired
perturbation-level bootstrap confidence interval for the main MSE difference.

Minimum factorial ablation:

| SAF | IAR | AMF | Purpose |
|---:|---:|---:|---|
| 0 | 0 | 0 | backbone |
| 1 | 0 | 0 | isolated SAF |
| 0 | 1 | 0 | isolated IAR |
| 0 | 0 | 1 | isolated AMF |
| 1 | 1 | 0 | SAF-IAR interaction |
| 1 | 0 | 1 | SAF-AMF interaction |
| 0 | 1 | 1 | IAR-AMF interaction |
| 1 | 1 | 1 | full model |

Support-ratio sensitivity must include 0.10, 0.15, 0.25, 0.40, and 1.00 and
report both all-gene and top-DE metrics.
