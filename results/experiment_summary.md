# FL-IDS Experiment Completion Summary

Generated: 2026-09-01

## Scope

The initial smoke runs used the two small CSV files first present in `data/raw/`:

- `Monday-WorkingHours.pcap_ISCX.csv`: 4,040 rows
- `Tuesday-WorkingHours.pcap_ISCX.csv`: 3,030 rows
- Cleaned dataset: 6,919 rows
- Train/test split: 5,535 train rows, 1,384 test rows
- Runtime device: CPU only

The earlier smoke runs validated the pipeline. Full CICIDS2017 CPU runs have now also been completed for the core centralized/federated comparison and loss ablation.

## Phase Status

| Phase | Status | Evidence |
|---|---|---|
| Phase 0: Environment & reproducibility | Complete for CPU runs | `requirements.txt`, `requirements-lock.txt`, deterministic seed utilities, successful PyTorch/PyG/Flower CPU runs |
| Phase 1: Diagnose imbalance | Complete for full data | `results/full/diagnostics_ce_d03_lr001/diagnose_metrics.json`, confusion-matrix PNGs |
| Phase 2: Class imbalance interventions | Core loss ablation complete for full data | class-weighted CE and focal loss full runs; random oversampling and stratified graph batching implemented and smoke-tested |
| Phase 3: Hyperparameter tuning | Smoke sweep complete | seeded dropout and LR grid in `results/ablation_hparams_seeded/` plus k sensitivity in `results/ablation_graphs_seeded/` |
| Phase 4: Centralized vs federated comparison | Complete for full data | `results/full/centralized_ce_d03_lr001/`, `results/full/federated_iid_ce_d03_lr001/`, `results/full/federated_non_iid_ce_d03_lr001/` |
| Phase 5: Paper update | Blocked | no paper source artifact exists in this workspace |
| Phase 6: Formatting & length | Blocked | no paper source artifact exists in this workspace |

## Full Dataset

All 8 CICIDS2017 MachineLearningCSV files are present in `data/raw/`.

| Stage | Rows / Shape |
|---|---:|
| Raw combined rows | 1,861,986 |
| Cleaned rows | 1,611,993 |
| Train rows | 1,289,594 |
| Test rows | 322,399 |
| Feature columns | 70 |
| Classes | 13 |
| Non-IID train graph batches | 646 |
| Test graph batches | 162 |

## Full-Data Results

Using CE, dropout 0.3, LR 0.01, weight decay 0.0005, k=10, seed 42:

| Setting | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Centralized | 0.9734 | 0.7295 | 0.7709 | 0.6979 | 0.9747 |
| Federated IID | 0.9593 | 0.5559 | 0.6762 | 0.5835 | 0.9613 |
| Federated non-IID | 0.9647 | 0.6335 | 0.6916 | 0.6215 | 0.9660 |

Full-data loss ablation, centralized:

| Loss | Macro-F1 | Weighted-F1 |
|---|---:|---:|
| CE | 0.6979 | 0.9747 |
| Class-weighted CE | 0.2788 | 0.6992 |
| Focal loss | 0.2845 | 0.7163 |

Best full-data per-class F1 from the federated IID run:

| Class | F1 |
|---|---:|
| BENIGN | 0.9732 |
| DDoS | 0.9640 |
| DoS GoldenEye | 0.9687 |
| DoS Hulk | 0.9546 |
| DoS Slowhttptest | 0.9198 |
| DoS slowloris | 0.8976 |
| PortScan | 0.8469 |
| Bot | 0.4960 |
| Heartbleed | 0.0000 |
| Infiltration | 0.1951 |
| Web Attack Brute Force | 0.3697 |
| Web Attack Sql Injection | 0.0000 |
| Web Attack XSS | 0.0000 |

The full-data run is much stronger than the smoke run. The remaining weaknesses are the rarest Web Attack classes, Heartbleed in the IID run, and SQL injection. Raw inverse-frequency weighting and default focal loss hurt macro-F1 and weighted-F1 overall.

## Baseline Diagnostic

| Setting | Macro-F1 | Weighted-F1 |
|---|---:|---:|
| Original centralized smoke baseline | 0.1375 | 0.5687 |
| Original federated non-IID smoke baseline | 0.1327 | 0.5565 |

Rare classes remain the failure mode. Diagnostic per-class F1:

| Class | F1 |
|---|---:|
| BENIGN | 0.8121 |
| Bot | 0.0000 |
| DDoS | 0.0000 |
| DoS Hulk | 0.0127 |
| FTP-Patator | 0.0000 |
| PortScan | 0.0000 |

## Imbalance Ablation

| Intervention | Macro-F1 | Weighted-F1 |
|---|---:|---:|
| CE baseline | 0.1371 | 0.5748 |
| Class-weighted CE | 0.1329 | 0.3716 |
| Focal loss | 0.0593 | 0.0772 |
| Random oversampling | 0.1107 | 0.3106 |
| Stratified graph batching | 0.1401 | 0.5728 |

Class-weighted CE, focal loss, and oversampling produced more nonzero minority-class F1 values, but on this smoke split they reduced overall macro-F1.

## Hyperparameter Sweep

Best centralized smoke setting:

| Loss | Dropout | LR | Weight Decay | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| CE | 0.3 | 0.01 | 0.0005 | 0.1484 | 0.5764 |

## k-NN Sensitivity

Using CE, dropout 0.3, LR 0.01, weight decay 0.0005:

| k | Macro-F1 | Weighted-F1 |
|---:|---:|---:|
| 5 | 0.1394 | 0.5761 |
| 10 | 0.1484 | 0.5764 |
| 20 | 0.1389 | 0.5740 |

## Final Smoke Comparison

Using CE, dropout 0.3, LR 0.01, weight decay 0.0005, seed 42:

| Setting | Macro-F1 | Weighted-F1 |
|---|---:|---:|
| Centralized | 0.1484 | 0.5764 |
| Federated IID | 0.1367 | 0.5730 |
| Federated non-IID | 0.1371 | 0.5748 |

## Remaining Work

The core full-data CPU experiments are complete. To fully complete the paper plan, update the manuscript tables and formatting once the paper source is available. For stronger final claims, run a larger GPU sweep over epochs, federated rounds, capped class weights, and focal-loss gamma.
