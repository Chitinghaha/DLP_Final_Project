# 6_6 CIFAR-100 Four-Coarse k20 Rebound Analysis

## 實驗設定

- Namespace：`sequential_four_coarse_no_flowers_cifar100_k20`
- Dataset：CIFAR-100
- Seed：`1`
- Target：4 個 coarse superclasses，每組 5 個 fine classes，共 20 個 fine classes。
- Significant rebound 定義：later accuracy `>= 10%`，且比 forget-step accuracy 高至少 `5pp`。

| Coarse superclass | Fine classes |
| --- | --- |
| 車輛 1 | 腳踏車 / bicycle (8), 公車 / bus (13), 摩托車 / motorcycle (48), 皮卡車 / pickup_truck (58), 火車 / train (90) |
| 大型自然場景 | 雲 / cloud (23), 森林 / forest (33), 山 / mountain (49), 平原 / plain (60), 海 / sea (71) |
| 大型雜食/草食動物 | 駱駝 / camel (15), 牛 / cattle (19), 黑猩猩 / chimpanzee (21), 大象 / elephant (31), 袋鼠 / kangaroo (38) |
| 大型人工戶外物 | 橋 / bridge (12), 城堡 / castle (17), 房子 / house (37), 道路 / road (68), 摩天大樓 / skyscraper (76) |

## Run summary

| Order | Run ID | Forget order | Progress | Eval CSVs | Final forget acc | Final retain acc | Final full test acc | Rebound rate | Final residual count | Missing eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clustered | four_coarse_clustered_seed1_k20 | 8,13,48,58,90,23,33,49,60,71,15,19,21,31,38,12,17,37,68,76 | missing | 0 | n/a | n/a | n/a | n/a | 0 | 20 |
| Interleaved | four_coarse_interleaved_seed1_k20 | 8,23,15,12,13,33,19,17,48,49,21,37,58,60,31,68,90,71,38,76 | missing | 0 | n/a | n/a | n/a | n/a | 0 | 20 |
| Block2 | four_coarse_block2_seed1_k20 | 8,13,23,33,15,19,12,17,48,58,49,60,21,31,37,68,90,71,38,76 | missing | 0 | n/a | n/a | n/a | n/a | 0 | 20 |

## 圖表

目前尚未找到完整 eval CSV，因此圖表會在實驗完成後重新產生。

## Rebound summary

目前尚未有 significant rebound，或實驗尚未完成。

## Coarse group summary

| Coarse group | Avg final target acc | Rebound count | Final residual count |
| --- | --- | --- | --- |
| 車輛 1 | n/a | 0 | 0 |
| 大型自然場景 | n/a | 0 | 0 |
| 大型雜食/草食動物 | n/a | 0 | 0 |
| 大型人工戶外物 | n/a | 0 | 0 |

## Tracked class trajectories

目前尚未有完整 trajectory；實驗完成後會列出 bicycle、chimpanzee、natural scenes、man-made outdoor 重點類別。

## 初步結論

- 實驗尚未完成；目前報告主要確認 target set、order、namespace 與分析輸出格式。

## Notes

- 本分析固定 `seed=1`，結論應視為機制探索，不主張跨 seed 普遍性。
- Accuracy 來自 CIFAR-100 test split；每個 fine class 有 100 張 test images，所以 1% 約等於 1 張圖。