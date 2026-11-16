# New tables: source-of-truth CSVs, five seeds


No old proxy or test-selected MACE result is included. Values are mean ± sample SD.


## Replacement Table 4: independent-calibration HGBR and ablation


| Configuration | AP | F1 | Retained / 10,000 | R² | MAE (eV/atom) |
|---|---|---|---|---|---|
| Clean reference | — | — | 10000.0 ± 0.0 | 0.6566 ± 0.0088 | 0.4222 ± 0.0105 |
| No cleaning | — | — | 10000.0 ± 0.0 | 0.5379 ± 0.0077 | 0.5154 ± 0.0110 |
| Statistical checks | 0.8010 ± 0.0089 | 0.8687 ± 0.0064 | 3638.2 ± 85.2 | 0.5682 ± 0.0155 | 0.4791 ± 0.0149 |
| Boundary checks | 0.8534 ± 0.0073 | 0.8695 ± 0.0064 | 3563.2 ± 80.9 | 0.5785 ± 0.0178 | 0.4714 ± 0.0149 |
| Complete rules | 0.9427 ± 0.0037 | 0.9705 ± 0.0019 | 4696.4 ± 20.9 | 0.6160 ± 0.0140 | 0.4495 ± 0.0132 |
| Rules + propagation | 0.9024 ± 0.0054 | 0.7715 ± 0.2776 | 6039.4 ± 1817.3 | 0.5828 ± 0.0433 | 0.4760 ± 0.0313 |
| Rules + GAT | 0.9516 ± 0.0039 | 0.9653 ± 0.0055 | 4755.2 ± 83.8 | 0.6137 ± 0.0145 | 0.4508 ± 0.0136 |
| Rules + propagation + GAT | 0.9530 ± 0.0041 | 0.9634 ± 0.0017 | 4706.8 ± 72.3 | 0.6107 ± 0.0143 | 0.4521 ± 0.0135 |
| Isolation Forest | 0.8687 ± 0.0233 | 0.7731 ± 0.0194 | 5264.0 ± 917.3 | 0.5403 ± 0.0198 | 0.5028 ± 0.0180 |
| PCA | 0.8144 ± 0.0042 | 0.7177 ± 0.0134 | 4748.2 ± 558.7 | 0.5067 ± 0.0183 | 0.5260 ± 0.0141 |
| Random (rule count) | — | — | 4696.4 ± 20.9 | 0.4773 ± 0.0122 | 0.5490 ± 0.0133 |
| Random (full count) | — | — | 4706.8 ± 72.3 | 0.4749 ± 0.0136 | 0.5492 ± 0.0113 |


## Paired HGBR R² differences


| Difference | Mean | 95% t interval |
|---|---|---|
| full_gat minus complete_rules | -0.0053 | [-0.0121, 0.0016] |
| bipartite minus complete_rules | -0.0332 | [-0.0882, 0.0218] |
| rules_gat minus complete_rules | -0.0023 | [-0.0093, 0.0047] |
| complete_rules minus no_cleaning | 0.0781 | [0.0683, 0.0880] |


## Runtime: three fresh processes per scale


| Actual n | Actual d | Total scoring s | Peak RSS MiB | Iterations |
|---|---|---|---|---|
| 999 | 43–46 | 0.787 ± 0.022 | 457.9 ± 27.7 | 39–39 |
| 3999 | 43–43 | 1.968 ± 0.004 | 468.0 ± 12.0 | 39–39 |
| 8000 | 43–43 | 3.684 ± 0.026 | 529.5 ± 0.7 | 38–39 |
| 15999 | 43–43 | 7.480 ± 0.176 | 699.1 ± 0.2 | 38–39 |


Total includes both 50-epoch GAT ablations; RSS includes interpreter, data and runtime. The 999-row repeat 1 retained 46 fields under its training-only eligibility rule; exact sizes and per-module times are disclosed in runtime_all_runs.csv.


## Replacement Table 10: historical scoring


| Pollution | Batch only | Historical KNN | Memory-GAT |
|---|---|---|---|
| 0% | 0.8637 ± 0.0054 | 0.8410 ± 0.0030 | 0.5912 ± 0.0663 |
| 10% | 0.8079 ± 0.0040 | 0.8266 ± 0.0034 | 0.5872 ± 0.0635 |
| 30% | 0.7294 ± 0.0022 | 0.7912 ± 0.0026 | 0.5776 ± 0.0583 |
| 50% | 0.6515 ± 0.0024 | 0.7581 ± 0.0020 | 0.5657 ± 0.0520 |


## Historical Memory-GAT cost: 900 current rows, 18 fields


| History nodes | Wall s | Peak process RSS MiB |
|---|---|---|
| 200 | 0.205 ± 0.002 | 388.4 ± 1.2 |
| 400 | 0.210 ± 0.002 | 389.0 ± 0.3 |
| 800 | 0.216 ± 0.002 | 390.1 ± 0.3 |


## Replacement Table 9: observable evidence and unresolved candidates


| Dataset / Excel row | Record ID | Evidence | Field / observed value | Status |
|---|---|---|---|---|
| alloy185 / 11 | Ti75Al10V5Cr5Nb5 | FE above 100 | FE: 130 | unusual value candidate |
| alloy185 / 15 | Ti67Al12V7Cr7Nb7 | FE above 100 | FE: 130 | unusual value candidate |
| alloy185 / 12 | Ti67Al12V7Cr7Nb7 | missing FE | FE: missing | field observation |
| alloy185 / 2 | Ti50V25Zr15Nb10 | missing UTS | UTS: missing | field observation |
| alloy185 / 23 | Fe50.0Mn30.0Co10.0Cr10.0 | composition property spread | UTS: 195.7 | context incomplete candidate |
| alloy315 / 193 | 192 | composition property spread | UTS: 378 | context incomplete candidate |


All entries have source_verified=False. Values for composition/property-spread events are the observed maximum-minus-minimum spread, in MPa for UTS or percentage points for FE. Missing fields are workbook observations; unusual FE and composition/property spreads require original-source and processing-context verification. Engineering elongation has no universal 100% ceiling.


## Penalty sensitivity (static propagation; independent calibration)


| Severe/mild | AP | F1 | Retained | R² |
|---|---|---|---|---|
| s0.2_m0.6 | 0.9052 ± 0.0052 | 0.7423 ± 0.3256 | 6174 ± 2040 | 0.5794 ± 0.0491 |
| s0.2_m0.7 | 0.9046 ± 0.0052 | 0.7619 ± 0.2766 | 6111 ± 1813 | 0.5803 ± 0.0385 |
| s0.2_m0.8 | 0.9038 ± 0.0053 | 0.7903 ± 0.2220 | 6001 ± 1567 | 0.5761 ± 0.0367 |
| s0.3_m0.6 | 0.9034 ± 0.0053 | 0.7498 ± 0.3392 | 6151 ± 2137 | 0.5855 ± 0.0438 |
| s0.3_m0.7 | 0.9024 ± 0.0054 | 0.7715 ± 0.2776 | 6039 ± 1817 | 0.5828 ± 0.0433 |
| s0.3_m0.8 | 0.9014 ± 0.0055 | 0.7960 ± 0.2179 | 5966 ± 1548 | 0.5781 ± 0.0351 |
| s0.4_m0.6 | 0.9010 ± 0.0056 | 0.7662 ± 0.4265 | 5866 ± 2318 | 0.5939 ± 0.0388 |
| s0.4_m0.7 | 0.9000 ± 0.0055 | 0.8318 ± 0.2797 | 5607 ± 1736 | 0.5970 ± 0.0353 |
| s0.4_m0.8 | 0.8991 ± 0.0056 | 0.8164 ± 0.2111 | 5823 ± 1508 | 0.5828 ± 0.0417 |


## Weight sensitivity (static propagation)


| Setting | AP | F1 | R² |
|---|---|---|---|
| accuracy_x0.0 | 0.9032 ± 0.0056 | 0.7851 ± 0.2689 | 0.5857 ± 0.0425 |
| accuracy_x0.8 | 0.9025 ± 0.0054 | 0.7726 ± 0.2766 | 0.5828 ± 0.0430 |
| accuracy_x1.2 | 0.9023 ± 0.0053 | 0.7683 ± 0.2807 | 0.5858 ± 0.0371 |
| dataset_characteristics_x0.0 | 0.9067 ± 0.0049 | 0.6881 ± 0.2692 | 0.5656 ± 0.0392 |
| dataset_characteristics_x0.8 | 0.9040 ± 0.0052 | 0.7657 ± 0.2816 | 0.5828 ± 0.0414 |
| dataset_characteristics_x1.2 | 0.9009 ± 0.0055 | 0.7837 ± 0.2726 | 0.5866 ± 0.0427 |
| default | 0.9024 ± 0.0054 | 0.7715 ± 0.2776 | 0.5828 ± 0.0433 |
| metadata_coupling_x0.0 | 0.9003 ± 0.0056 | 0.7827 ± 0.2714 | 0.5849 ± 0.0402 |
| metadata_coupling_x0.8 | 0.9020 ± 0.0054 | 0.7720 ± 0.2776 | 0.5836 ± 0.0434 |
| metadata_coupling_x1.2 | 0.9028 ± 0.0053 | 0.7688 ± 0.2798 | 0.5826 ± 0.0425 |
| non_duplicate_rows_x0.0 | 0.9011 ± 0.0055 | 0.7813 ± 0.2715 | 0.5835 ± 0.0402 |
| non_duplicate_rows_x0.8 | 0.9021 ± 0.0054 | 0.7716 ± 0.2777 | 0.5851 ± 0.0413 |
| non_duplicate_rows_x1.2 | 0.9027 ± 0.0054 | 0.7713 ± 0.2777 | 0.5823 ± 0.0429 |
| non_missing_cells_x0.0 | 0.8980 ± 0.0058 | 0.8380 ± 0.2657 | 0.5976 ± 0.0357 |
| non_missing_cells_x0.8 | 0.9014 ± 0.0055 | 0.7811 ± 0.2713 | 0.5839 ± 0.0376 |
| non_missing_cells_x1.2 | 0.9034 ± 0.0053 | 0.7675 ± 0.2809 | 0.5827 ± 0.0420 |
| provenance_x0.0 | 0.9011 ± 0.0055 | 0.7813 ± 0.2715 | 0.5835 ± 0.0402 |
| provenance_x0.8 | 0.9021 ± 0.0054 | 0.7716 ± 0.2777 | 0.5851 ± 0.0413 |
| provenance_x1.2 | 0.9027 ± 0.0054 | 0.7713 ± 0.2777 | 0.5823 ± 0.0429 |
| target_correlation_x0.0 | 0.9038 ± 0.0052 | 0.8434 ± 0.2550 | 0.5977 ± 0.0330 |
| target_correlation_x0.8 | 0.9027 ± 0.0053 | 0.7797 ± 0.2703 | 0.5848 ± 0.0414 |
| target_correlation_x1.2 | 0.9021 ± 0.0055 | 0.7686 ± 0.2825 | 0.5817 ± 0.0422 |
| un_skewness_x0.0 | 0.9052 ± 0.0051 | 0.7435 ± 0.3276 | 0.5803 ± 0.0496 |
| un_skewness_x0.8 | 0.9031 ± 0.0054 | 0.7649 ± 0.2870 | 0.5821 ± 0.0414 |
| un_skewness_x1.2 | 0.9017 ± 0.0054 | 0.7817 ± 0.2685 | 0.5834 ± 0.0415 |
| uniform | 0.9047 ± 0.0051 | 0.7615 ± 0.2786 | 0.5807 ± 0.0396 |
| uniformity_x0.0 | 0.8988 ± 0.0056 | 0.7900 ± 0.2689 | 0.5863 ± 0.0443 |
| uniformity_x0.8 | 0.9016 ± 0.0055 | 0.7810 ± 0.2713 | 0.5835 ± 0.0394 |
| uniformity_x1.2 | 0.9032 ± 0.0053 | 0.7679 ± 0.2805 | 0.5823 ± 0.0418 |


## Replacement Table 5: corruption fraction × mixed-input scale


| Fraction | Scale | Method | F1 | R² |
|---|---|---|---|---|
| 10% | 0.1 | Rules + propagation | 0.4637 ± 0.4233 | 0.6349 ± 0.0190 |
| 10% | 0.1 | Complete rules | 0.7815 ± 0.0102 | 0.6467 ± 0.0086 |
| 10% | 0.1 | Rules + propagation + GAT | 0.7327 ± 0.0160 | 0.6416 ± 0.0141 |
| 10% | 0.1 | No cleaning | — | 0.6165 ± 0.0059 |
| 10% | 0.1 | Rules + GAT | 0.7779 ± 0.0107 | 0.6462 ± 0.0071 |
| 10% | 0.3 | Rules + propagation | 0.4624 ± 0.4203 | 0.6358 ± 0.0193 |
| 10% | 0.3 | Complete rules | 0.7815 ± 0.0102 | 0.6467 ± 0.0086 |
| 10% | 0.3 | Rules + propagation + GAT | 0.7279 ± 0.0212 | 0.6402 ± 0.0119 |
| 10% | 0.3 | No cleaning | — | 0.6169 ± 0.0081 |
| 10% | 0.3 | Rules + GAT | 0.7782 ± 0.0105 | 0.6455 ± 0.0073 |
| 10% | 0.6 | Rules + propagation | 0.5008 ± 0.3770 | 0.6338 ± 0.0211 |
| 10% | 0.6 | Complete rules | 0.7816 ± 0.0101 | 0.6469 ± 0.0082 |
| 10% | 0.6 | Rules + propagation + GAT | 0.7278 ± 0.0206 | 0.6385 ± 0.0137 |
| 10% | 0.6 | No cleaning | — | 0.6130 ± 0.0085 |
| 10% | 0.6 | Rules + GAT | 0.7785 ± 0.0107 | 0.6438 ± 0.0096 |
| 30% | 0.1 | Rules + propagation | 0.5956 ± 0.3827 | 0.6025 ± 0.0344 |
| 30% | 0.1 | Complete rules | 0.9310 ± 0.0024 | 0.6375 ± 0.0105 |
| 30% | 0.1 | Rules + propagation + GAT | 0.9048 ± 0.0296 | 0.6252 ± 0.0283 |
| 30% | 0.1 | No cleaning | — | 0.5705 ± 0.0109 |
| 30% | 0.1 | Rules + GAT | 0.9249 ± 0.0097 | 0.6337 ± 0.0188 |
| 30% | 0.3 | Rules + propagation | 0.6727 ± 0.3955 | 0.6101 ± 0.0359 |
| 30% | 0.3 | Complete rules | 0.9310 ± 0.0024 | 0.6367 ± 0.0115 |
| 30% | 0.3 | Rules + propagation + GAT | 0.9108 ± 0.0183 | 0.6271 ± 0.0188 |
| 30% | 0.3 | No cleaning | — | 0.5733 ± 0.0116 |
| 30% | 0.3 | Rules + GAT | 0.9252 ± 0.0077 | 0.6327 ± 0.0191 |
| 30% | 0.6 | Rules + propagation | 0.7385 ± 0.4129 | 0.6240 ± 0.0354 |
| 30% | 0.6 | Complete rules | 0.9310 ± 0.0024 | 0.6375 ± 0.0105 |
| 30% | 0.6 | Rules + propagation + GAT | 0.9103 ± 0.0189 | 0.6293 ± 0.0164 |
| 30% | 0.6 | No cleaning | — | 0.5720 ± 0.0126 |
| 30% | 0.6 | Rules + GAT | 0.9254 ± 0.0071 | 0.6331 ± 0.0162 |
| 50% | 0.1 | Rules + propagation | 0.6999 ± 0.3958 | 0.5798 ± 0.0339 |
| 50% | 0.1 | Complete rules | 0.9705 ± 0.0020 | 0.6171 ± 0.0139 |
| 50% | 0.1 | Rules + propagation + GAT | 0.9615 ± 0.0012 | 0.6123 ± 0.0168 |
| 50% | 0.1 | No cleaning | — | 0.5367 ± 0.0076 |
| 50% | 0.1 | Rules + GAT | 0.9661 ± 0.0039 | 0.6148 ± 0.0136 |
| 50% | 0.3 | Rules + propagation | 0.6872 ± 0.2916 | 0.5702 ± 0.0346 |
| 50% | 0.3 | Complete rules | 0.9705 ± 0.0019 | 0.6163 ± 0.0144 |
| 50% | 0.3 | Rules + propagation + GAT | 0.9632 ± 0.0022 | 0.6127 ± 0.0175 |
| 50% | 0.3 | No cleaning | — | 0.5350 ± 0.0069 |
| 50% | 0.3 | Rules + GAT | 0.9656 ± 0.0057 | 0.6155 ± 0.0142 |
| 50% | 0.6 | Rules + propagation | 0.7715 ± 0.2776 | 0.5828 ± 0.0433 |
| 50% | 0.6 | Complete rules | 0.9705 ± 0.0019 | 0.6160 ± 0.0140 |
| 50% | 0.6 | Rules + propagation + GAT | 0.9634 ± 0.0017 | 0.6107 ± 0.0143 |
| 50% | 0.6 | No cleaning | — | 0.5379 ± 0.0077 |
| 50% | 0.6 | Rules + GAT | 0.9653 ± 0.0055 | 0.6137 ± 0.0145 |


## Replacement Table 6: per-corruption recall


| Method | Injected type | Recall |
|---|---|---|
| Rules + propagation | domain | 0.9832 ± 0.0371 |
| Rules + propagation | mixed | 0.7053 ± 0.4073 |
| Rules + propagation | relationship | 0.6000 ± 0.5477 |
| Rules + propagation | statistical | 0.6373 ± 0.5006 |
| Boundary checks | domain | 1.0000 ± 0.0000 |
| Boundary checks | mixed | 0.9974 ± 0.0016 |
| Boundary checks | relationship | 0.9811 ± 0.0037 |
| Boundary checks | statistical | 0.9992 ± 0.0006 |
| Complete rules | domain | 1.0000 ± 0.0000 |
| Complete rules | mixed | 0.9998 ± 0.0004 |
| Complete rules | relationship | 1.0000 ± 0.0000 |
| Complete rules | statistical | 1.0000 ± 0.0000 |
| Rules + propagation + GAT | domain | 0.9990 ± 0.0010 |
| Rules + propagation + GAT | mixed | 0.9997 ± 0.0004 |
| Rules + propagation + GAT | relationship | 0.9995 ± 0.0007 |
| Rules + propagation + GAT | statistical | 0.9685 ± 0.0268 |
| Rules + GAT | domain | 0.9926 ± 0.0121 |
| Rules + GAT | mixed | 0.9984 ± 0.0020 |
| Rules + GAT | relationship | 0.9986 ± 0.0020 |
| Rules + GAT | statistical | 0.9662 ± 0.0370 |
| Statistical checks | domain | 0.9979 ± 0.0018 |
| Statistical checks | mixed | 0.9706 ± 0.0068 |
| Statistical checks | relationship | 0.9803 ± 0.0032 |
| Statistical checks | statistical | 0.9992 ± 0.0006 |


## Replacement Tables 7–8: true-clean / injected groups


| Method | Clean confidence | Injected confidence | Gap |
|---|---|---|---|
| Rules + propagation | 0.7709 ± 0.0035 | 0.5686 ± 0.0013 | 0.2023 ± 0.0023 |
| Boundary checks | 0.8878 ± 0.0064 | 0.4598 ± 0.0011 | 0.4281 ± 0.0069 |
| Complete rules | 0.8878 ± 0.0064 | 0.3000 ± 0.0000 | 0.5878 ± 0.0064 |
| Rules + propagation + GAT | 0.6483 ± 0.0074 | 0.4215 ± 0.0053 | 0.2268 ± 0.0116 |
| Rules + GAT | 0.7477 ± 0.0101 | 0.2238 ± 0.0029 | 0.5239 ± 0.0121 |
| Statistical checks | 0.9133 ± 0.0050 | 0.6477 ± 0.0012 | 0.2656 ± 0.0057 |


Groups use controlled injection labels only for analysis. Confidence means are on stage-specific scales and are not probabilities.


## Replacement Table 3: repeated grouped MACE (4,000-row subsets)


| Condition | F1 after common safety | R² | MAE eV/atom | Retained |
|---|---|---|---|---|
| Clean reference | — | 0.9387 ± 0.0158 | 0.1488 ± 0.0195 | 2256.0 ± 22.5 |
| No cleaning | 0.0000 ± 0.0000 | 0.5896 ± 0.0527 | 0.4908 ± 0.0248 | 2256.0 ± 22.5 |
| Complete rules | 0.9622 ± 0.0039 | 0.8801 ± 0.0268 | 0.2121 ± 0.0316 | 1157.8 ± 8.0 |
| Rules + propagation + GAT | 0.9451 ± 0.0119 | 0.8150 ± 0.0413 | 0.2790 ± 0.0328 | 1184.6 ± 49.1 |
| Isolation Forest | 0.7002 ± 0.0916 | 0.6589 ± 0.0602 | 0.4335 ± 0.0257 | 1350.2 ± 393.6 |
| PCA | 0.6429 ± 0.0256 | 0.5740 ± 0.0576 | 0.4963 ± 0.0304 | 1466.2 ± 146.5 |
| Matched random | 0.4761 ± 0.0124 | 0.2720 ± 0.1468 | 0.6444 ± 0.0541 | 1157.8 ± 8.0 |


## Paired MACE R² differences


| Difference | Mean | 95% t interval |
|---|---|---|
| complete_rules minus no_cleaning | 0.2906 | [0.2098, 0.3714] |
| full_gat minus complete_rules | -0.0652 | [-0.1321, 0.0018] |