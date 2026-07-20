# Table 2 — LOP7 Energy Measurement Results (macOS / Apple Silicon)

Generated: 2026-07-20 21:44:49  
Statistics: decoder 10 run(s), CPU 10 run(s)

| Measurement | Condition | Runtime (ms) | Net energy (J) |
|---|---|---|---|
| NNVC decoder | LOP7 off, CPU | 115.23 ± 1.98 | 0.761 ± 0.214 |
| NNVC decoder | LOP7 on, CPU | 563.80 ± 2.64 | 3.742 ± 0.262 |
| NNVC decoder | Overhead (on − off) | 448.56 ± 3.30 | 2.981 ± 0.338 |
| Isolated LOP7 | Single patch, CPU | 3.58 ± 0.03 | 0.019 ± 0.006 |
| Calculated | 24× single patch, CPU | 85.81 ± 0.82 | 0.456 ± 0.143 |
| Comparison | Overhead / 24× CPU | 5.23× | 6.54× |
