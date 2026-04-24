# Classification 實驗總覽導航

## 1. 總覽摘要

這份文件是 `Unlearn-Saliency-original/Classification/` 目前正式實驗的總入口，目的是回答三個問題：

1. 目前做過哪些正式實驗？
2. 每條實驗線各自在解什麼問題？
3. 如果要理解目前進度，應該先看哪份報告？

目前可以把正式實驗分成 5 條主線：

- `original baseline`
- 舊版 `sequential`
- 修正版 `sequential_cumulative`
- 修正版 `sequential_incremental`
- `sequential_incremental_ordered`

這 5 條主線之間的關係很重要：

- `original baseline` 是所有 sequential forgetting 實驗的比較基準。
- 舊版 `sequential/full_seed1_k9` 暴露出一個核心問題：新 class 當步可以忘掉，但由於資料流程設計沒有把已 forgotten data 永久排除，舊 forgotten classes 會在後續 step 中恢復。
- `sequential_cumulative` 與 `sequential_incremental` 是針對這個問題提出的修正版。
- `sequential_incremental_ordered` 則是在正確 incremental data flow 下，進一步測試不同 forget order 會不會改變 forgetting 難度與穩定性；目前 `clustered_animals` 與 `interleaved` 都已補齊完整 trajectory。
- 先前做過的 `Ordered Sequential Overnight` 因資料流程錯誤，會把舊 forgotten data 帶回後續訓練，所以不納入目前正式結論；詳見 [ordered_sequential_overnight_20260424_analysis.md](./ordered_sequential_overnight_20260424_analysis.md) 的作廢說明。

目前主線也可以整理成更清楚的分工：

- `full_seed1_k9` 主要回答「單一新 class 的當步 forgetting 本身有沒有效」。
- `sequential_experiment_comparison` 主要回答「修正資料流程後，能否避免舊 forgotten classes recovery」。
- `sequential_incremental_ordered` 主要回答「在 recovery 被排除之後，連續 forgetting 還會不會出現 order-dependent lag」。

如果你的目標是理解「`UNLEARN` 什麼時候會失效」，目前最重要的觀察來自：

- [full_seed1_k9_analysis.md](./full_seed1_k9_analysis.md)
- [sequential_experiment_comparison.md](./sequential_experiment_comparison.md)
- [sequential_incremental_ordered_analysis.md](./sequential_incremental_ordered_analysis.md)

## 2. 目前正式實驗地圖

| Experiment family | Representative run(s) | 核心問題 | 目前狀態 | 關鍵結論 | 對應報告 / log |
| --- | --- | --- | --- | --- | --- |
| `original baseline` | `seed1_baseline`、原始 model train/eval | 原始模型表現是多少？後續 forgetting 要和誰比較？ | 已完成 | 提供 CIFAR-10 / ResNet-18 的 baseline accuracy 與 per-class 參考 | [README.md](./README.md)、`results/eval/original/seed1_baseline.csv`、`results/logs/original/` |
| 舊版 `sequential` | `full_seed1_k9` | 直接做 sequential forgetting 會發生什麼事？ | 已完成，且已有分析稿 | 新 class 可在當步被忘掉，但已 forgotten data 在後續步驟又回到訓練流程，造成舊 forgotten classes recovery / relearning | [full_seed1_k9_analysis.md](./full_seed1_k9_analysis.md)、`results/logs/sequential/full_seed1_k9/` |
| `sequential_cumulative` | `cumulative_mask_seed1_k9` | 若每一步都持續對累積 forgotten classes 施壓，能否避免 recovery？ | 已完成，且已有比較稿 | 中後期 forgetting 維持最乾淨，但成本較高 | [sequential_experiment_comparison.md](./sequential_experiment_comparison.md)、`results/logs/sequential_cumulative/cumulative_mask_seed1_k9/` |
| `sequential_incremental` | `incremental_newclass_seed1_k9` | 若只忘新 class、同時完全排除舊 forgotten data，能否避免 recovery？ | 已完成，且已有比較稿 | 可大幅改善舊 class recovery，且效率最佳，適合當主線候選 | [sequential_experiment_comparison.md](./sequential_experiment_comparison.md)、`results/logs/sequential_incremental/incremental_newclass_seed1_k9/` |
| `sequential_incremental_ordered` | `incremental_clustered_animals_seed1_k9`、`incremental_interleaved_seed1_k9`、`incremental_interleaved_seed1_k9_resume_from_step6` | 在正確 incremental data flow 下，單純改變 forget order 是否仍會影響 forgetting 難度與穩定性？ | 已完成，且已有完整分析稿 | 兩條 ordered 線 final 都能收斂到完整 forgetting；差別主要不在 final endpoint，而在 immediate forgetting lag 出現在哪些 step、持續多久 | [sequential_incremental_ordered_analysis.md](./sequential_incremental_ordered_analysis.md)、`results/logs/sequential_incremental_ordered/`、`results/logs/sequential_incremental_ordered_resume/` |

## 3. 各實驗線整理

### 3.1 `original baseline`

- 這條線在測什麼  
  建立所有 forgetting 實驗的原始參考點，包括 full test accuracy 與 per-class accuracy。
- 代表性 run / seed  
  `results/eval/original/seed1_baseline.csv`，對應原始模型 `results/original/cifar10_resnet18_seed1/0model_SA_best.pth.tar`。
- 已知主要結果  
  baseline 是後續判斷 retain class 是否受損、以及比較不同 sequential 設計的重要基準。
- 目前限制  
  目前主要分析稿引用的是 seed 1 baseline；seed 2 雖有 train log，但沒有被完整整理成獨立報告。
- 應該回看哪份報告  
  先看 [README.md](./README.md) 確認訓練與 unlearning 指令，再看 `seed1_baseline.csv`。

### 3.2 舊版 `sequential`

- 這條線在測什麼  
  直接把 sequential forgetting 接在前一步 checkpoint 後面跑，觀察 9-step cumulative forgetting 的實際行為。
- 代表性 run / seed  
  `results/logs/sequential/full_seed1_k9/`，seed 1。
- 已知主要結果  
  每一步新 forgotten class 當下通常能被壓低，但由於實驗設計沒有把已 forgotten data 永久排除在後續訓練之外，舊 forgotten classes 會在後續 step 中恢復，導致 cumulative `forget_accuracy` 上升、`UA` 下降。
- 目前限制  
  這條線最重要的價值是暴露 failure mode，不適合被視為最終方法候選。
- 應該回看哪份報告  
  [full_seed1_k9_analysis.md](./full_seed1_k9_analysis.md)

### 3.3 `sequential_cumulative`

- 這條線在測什麼  
  每一步都把目前累積 forgotten classes 納入 forget optimization，測試是否能持續壓制舊 forgotten classes。
- 代表性 run / seed  
  `results/logs/sequential_cumulative/cumulative_mask_seed1_k9/`，seed 1。
- 已知主要結果  
  對 cumulative forgetting 的維持最強，中後期 `forget_accuracy` 更容易接近 `0`，但 runtime 較長。
- 目前限制  
  雖然 forgetting 很乾淨，但成本比 incremental 高，也更不像「只忘新 class」的問題設定。
- 應該回看哪份報告  
  [sequential_experiment_comparison.md](./sequential_experiment_comparison.md)

### 3.4 `sequential_incremental`

- 這條線在測什麼  
  當步只對新 class 做 forget training，同時完全排除舊 forgotten classes，不讓它們再進後續 train loader。
- 代表性 run / seed  
  `results/logs/sequential_incremental/incremental_newclass_seed1_k9/`，seed 1。
- 已知主要結果  
  可以明顯避免舊 forgotten classes recovery，且在目前已比較的方法裡效率最佳，是最接近主線方法的候選。
- 目前限制  
  雖然比舊版 sequential 好很多，但從 ordered 報告來看，正確 data flow 並不保證每一步都能立刻忘乾淨；中後段仍可能出現 immediate forgetting lag，且目前仍缺多 seed 驗證。
- 應該回看哪份報告  
  [sequential_experiment_comparison.md](./sequential_experiment_comparison.md)

### 3.5 `sequential_incremental_ordered`

- 這條線在測什麼  
  在正確 incremental data flow 下，只改變 forget order，觀察不同 order 是否仍會改變 forgetting 的收斂難度、穩定性與 runtime。
- 代表性 run / seed  
  `results/logs/sequential_incremental_ordered/incremental_clustered_animals_seed1_k9/`、`results/logs/sequential_incremental_ordered/incremental_interleaved_seed1_k9/` 與 `results/logs/sequential_incremental_ordered_resume/incremental_interleaved_seed1_k9_resume_from_step6/`，seed 1。
- 已知主要結果  
  `clustered_animals` 與 stitched `interleaved` final 都能達到完整 forgetting，但兩者前中後段的 immediate forgetting lag 分布不同。`clustered_animals` 較像前段短暫 lag、後段快速補忘；`interleaved` 則在前中段更常出現 lag，但後續仍會逐步收斂。
- 目前限制  
  目前仍只有 seed 1，且 order 樣本數有限，因此不能把 `clustered_animals` 與 `interleaved` 的差異直接一般化成所有 forget order 的普遍排名。
- 應該回看哪份報告  
  [sequential_incremental_ordered_analysis.md](./sequential_incremental_ordered_analysis.md)

## 4. 目前已知結論

目前主要分析稿串起來，可以得到以下一致結論：

1. baseline 是所有 forgetting 實驗的比較基準。  
   沒有 baseline，就很難判斷 retain class 是否真的被傷到。

2. 舊版 `sequential/full_seed1_k9` 已經清楚暴露核心問題。  
   問題不是「當步新 class 忘不掉」，而是資料流程設計把已 forgotten data 又帶回後續訓練，導致舊 forgotten classes 在後續 step 中恢復。

3. `sequential_experiment_comparison.md` 證明修正版方法有效。  
   `cumulative` 與 `incremental` 都大幅改善舊 forgotten class recovery，其中 `incremental` 更符合問題定義且更快。

4. 把 `full_seed1_k9` 與 `sequential_incremental_ordered` 合起來看，可以把連續 forgetting 的兩種困難分開。  
   前者說明單一新 class 的當步 forgetting 通常有效，但舊版流程會造成 early forgotten classes recovery；後者則顯示在 recovery 被排除後，連續 forgetting 仍可能出現 immediate forgetting lag。

5. `sequential_incremental_ordered` 顯示：即使資料流程正確，order 仍可能影響 forgetting 的難度與穩定性。  
   兩條 ordered 線 final 都能到同一個完整 forgetting 終點，但 lag 出現在哪些 step、需要幾步才收斂，仍會隨 order 改變。

6. 如果要找目前最像主線候選的方法，應優先看 `sequential_incremental`。  
   如果要研究 forgetting 何時失效，則應優先對照舊版 `sequential` 的 recovery failure case，再看 ordered / incremental 線中的 lag 現象。

## 5. 目前還沒解決的問題

目前最重要但尚未被完全系統化整理的問題有三個：

1. `UNLEARN` 何時開始失效，還沒有被整理成 failure condition 表格。  
   我們現在已經知道失效型態至少有兩種：資料流程錯誤時的 recovery，以及正確 incremental 流程下的 immediate forgetting lag；但還沒有系統化列出：
   - 哪一個 class 在哪一步被 forget
   - 之後是在後續 step 中 recovery，還是先出現 lag 再逐步收斂
   - 最終回升或收斂到多少

2. seed effect 尚未充分驗證。  
   目前主線比較仍以 seed 1 為主，多 seed 結論還不夠完整。

3. order effect 的一般化仍不足。  
   雖然 `clustered_animals` 與 `interleaved` 都已補齊，但目前 order 樣本仍很少，還不能對所有 possible order 做穩定排名。

## 6. 建議閱讀順序

如果你現在想快速進入狀況，建議依照下面順序閱讀：

1. 先看 baseline / overall setting  
   先讀 [README.md](./README.md)，再看 `results/eval/original/seed1_baseline.csv`，建立共同背景。

2. 再看舊版失效案例  
   讀 [full_seed1_k9_analysis.md](./full_seed1_k9_analysis.md)，理解已 forgotten data 被重新帶回後續訓練時，舊 forgotten classes 的 recovery 是怎麼發生的。

3. 再看修正版方法比較  
   讀 [sequential_experiment_comparison.md](./sequential_experiment_comparison.md)，確認 `cumulative` / `incremental` 如何改善舊版問題。

4. 再看最新的 incremental ordered 結果  
   讀 [sequential_incremental_ordered_analysis.md](./sequential_incremental_ordered_analysis.md)，理解在正確 data flow 下，為什麼 `clustered_animals` 與 `interleaved` 都能 final forget，但仍會出現不同型態的 immediate forgetting lag。

5. 若需要追溯作廢實驗，再看作廢說明  
   讀 [ordered_sequential_overnight_20260424_analysis.md](./ordered_sequential_overnight_20260424_analysis.md)，理解為何先前的 ordered overnight 結果不再納入正式討論。

## 7. 附錄：未納入主線的 smoke / 測試 run

下表列出目前存在、但未納入正式主線結論的 smoke / 測試性 run。

| Run 名稱 | 類型 | 用途 | 是否納入正式結論 |
| --- | --- | --- | --- |
| `smoke_noop` | sequential smoke | 檢查 pipeline 是否能空跑或基本啟動 | 否 |
| `smoke_seed99_k1` | sequential smoke | 用最小 step 驗證 sequential 流程 | 否 |
| `smoke_seed99_k1_py_path` | sequential smoke | 驗證 Python path / 執行環境設定 | 否 |
| `smoke_cumulative_mask_seed1_k2` | cumulative smoke | 驗證 cumulative mask 版本前兩步流程 | 否 |
| `smoke_newclass_mask_seed1_k2` | cumulative-side smoke | 驗證新 class mask 相關流程 | 否 |
| `smoke_incremental_newclass_seed1_k2` | incremental smoke | 驗證 incremental 版本前兩步流程 | 否 |
| `smoke_clustered_animals_seed1_k2` | ordered smoke | 早期 ordered overnight smoke，因資料流程錯誤不納入 | 否 |
| `smoke_clustered_animals_seed1_k2_manual` | ordered manual smoke | 手動重跑 clustered smoke 作檢查 | 否 |
| `smoke_interleaved_seed1_k2` | ordered smoke | 早期 ordered overnight smoke，因資料流程錯誤不納入 | 否 |
| `smoke_retain_bird_seed1_k2` | ordered smoke | 早期 ordered overnight smoke，因資料流程錯誤不納入 | 否 |
| `overnight_20260424_015845` | ordered queue log | 早期 overnight 啟動紀錄 | 否 |
| `overnight_20260424_015914` | ordered queue log | 早期 overnight queue 紀錄 | 否 |

如果只是想理解「目前研究做到哪裡」，可以先完全忽略這些附錄項目，直接看主文中的 5 條正式實驗線。
