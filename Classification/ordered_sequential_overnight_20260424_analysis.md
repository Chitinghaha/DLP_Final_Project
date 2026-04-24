# Ordered Sequential Overnight 實驗作廢說明

這份文件不再是正式分析稿，而是作為 `Ordered Sequential Overnight` 這批實驗的作廢註記。

## 作廢範圍

本說明對應的實驗範圍是：

- result namespace: `sequential_ordered`
- queue log: `results/logs/sequential_ordered/overnight_20260424_020020/queue.log`
- 主要 run：
  - `canonical_seed2_k9`
  - `clustered_animals_seed1_k9`
  - `clustered_animals_seed2_k9`
  - `interleaved_seed1_k9`
  - `retain_bird_seed1_k9`

## 作廢原因

這批 ordered overnight 實驗的資料流程設計有錯誤：

- 每一步只指定當步新的 `class_to_replace`
- 舊 forgotten classes 沒有像 `sequential_incremental` 那樣被永久排除出後續 train loader
- 因此舊 forgotten data 會在後續步驟重新回到 train / retain 流程

這代表這批實驗不適合被用來討論：

- 不同 forget order 的有效性排名
- forgetting / retain / runtime 的正式 trade-off 結論
- `UNLEARN` failure condition 的正式證據

## 哪些原結論不可再採用

以下類型的結論，現在都不再納入正式研究敘事：

- 哪個 order forgetting 最強
- 哪個 order 最 balanced
- 哪個 order runtime 最有效率
- 任何把這批 `sequential_ordered` 結果拿來和 `sequential_incremental` / `sequential_cumulative` 正式對比的說法

原因不是這些數字不存在，而是它們建立在錯誤的資料流程上，因此不再具有研究解釋力。

## 正確替代線

如果要討論目前有效的主線結果，請優先回看：

- [full_seed1_k9_analysis.md](./full_seed1_k9_analysis.md)
  - 用來理解舊版 sequential failure case
- [sequential_experiment_comparison.md](./sequential_experiment_comparison.md)
  - 用來比較 `sequential_cumulative` 與 `sequential_incremental`

如果要研究「在正確 incremental data flow 下，單純改變 forget order 會發生什麼事」，應改看後續新的 `sequential_incremental_ordered` 實驗線，而不是這批已作廢的 `sequential_ordered` overnight 結果。
