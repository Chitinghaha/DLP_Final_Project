# CIFAR-100 Single-Coarse Rebound Analysis

## Summary

- Namespace: `sequential_single_coarse_rebound_cifar100`
- Completed runs: `9/9`
- Significant rebound rows: `3`
- Final residual rows: `3`
- Significant rebound definition: `max_later_accuracy >= 10%` and at least `+5pp` above accuracy at forget step.
- This experiment fixes `seed=1`; it does not claim cross-seed generality.

## Run Summary

| Group | Order | Run ID | Forget order | Progress | Eval CSVs | Final forget acc | Final retain acc | Rebound rate | Final residual count | Missing eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 大型雜食/草食動物 | official | large_omni_official_seed1_k5 | 15,19,21,31,38 | success | 5 | 0.2 | 70.6 | 0/4 | 0 | 0 |
| 大型雜食/草食動物 | random | large_omni_random_seed1_k5 | 21,15,38,19,31 | success | 5 | 0.6 | 70.5 | 0/4 | 0 | 0 |
| 大型雜食/草食動物 | hardfirst | large_omni_hardfirst_seed1_k5 | 19,38,31,15,21 | success | 5 | 0.0 | 70.6 | 0/4 | 0 | 0 |
| 車輛 1 | official | vehicles1_official_seed1_k5 | 8,13,48,58,90 | success | 5 | 8.6 | 70.1 | 1/4 | 1 | 0 |
| 車輛 1 | random | vehicles1_random_seed1_k5 | 48,8,90,13,58 | success | 5 | 8.0 | 70.3 | 1/4 | 1 | 0 |
| 車輛 1 | hardfirst | vehicles1_hardfirst_seed1_k5 | 13,8,58,90,48 | success | 5 | 9.6 | 70.3 | 1/4 | 1 | 0 |
| 花卉 | official | flowers_official_seed1_k5 | 54,62,70,82,92 | success | 5 | 0.0 | 70.3 | 0/4 | 0 | 0 |
| 花卉 | random | flowers_random_seed1_k5 | 70,54,92,62,82 | success | 5 | 0.0 | 70.3 | 0/4 | 0 | 0 |
| 花卉 | hardfirst | flowers_hardfirst_seed1_k5 | 92,70,62,54,82 | success | 5 | 0.0 | 70.5 | 0/4 | 0 | 0 |

## Significant Post-Forget Rebound

| Group | Order | Class | Forget step | At forget | Max later | Final |
| --- | --- | --- | --- | --- | --- | --- |
| 車輛 1 | official | 腳踏車 / bicycle (8) | 1 | 0.0 | 43.0 | 43.0 |
| 車輛 1 | random | 腳踏車 / bicycle (8) | 2 | 2.0 | 41.0 | 40.0 |
| 車輛 1 | hardfirst | 腳踏車 / bicycle (8) | 2 | 1.0 | 48.0 | 48.0 |

## Immediate Forgetting Lag

| Group | Order | Class | Forget step | At forget |
| --- | --- | --- | --- | --- |
| none | - | - | - | - |

## Final Residual

| Group | Order | Class | Final accuracy |
| --- | --- | --- | --- |
| 車輛 1 | official | 腳踏車 / bicycle (8) | 43.0 |
| 車輛 1 | random | 腳踏車 / bicycle (8) | 40.0 |
| 車輛 1 | hardfirst | 腳踏車 / bicycle (8) | 48.0 |

## Class 21 Check

Class 21 (`chimpanzee / 黑猩猩`) is included only in the large omnivores/herbivores runs.

| Order | Forget step | At forget | Max later | Final | Significant rebound |
| --- | --- | --- | --- | --- | --- |
| official | 3 | 0.0 | 1.0 | 1.0 | no |
| random | 1 | 0.0 | 4.0 | 3.0 | no |
| hardfirst | 5 | 0.0 | n/a | 0.0 | no |
