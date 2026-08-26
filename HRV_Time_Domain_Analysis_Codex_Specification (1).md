# HRV Time-Domain Analysis — Codex Implementation Specification

## 0. Purpose

本文件是提供給 Codex 的**實作規格（Specification）**。

Codex 的任務是依照本文件實作 HRV time-domain analysis pipeline。

### 強制治理規則

1. **不得自行決定研究方法學。**
2. 本文件未明確定義的分析方法、統計方法、圖表規則、資料清理規則、缺失值處理規則，不得自行新增。
3. 若程式實作必須知道某項未定義設定，應：
   - 暫停該部分實作，或
   - 明確標記為 `TODO / REQUIRES_HUMAN_DECISION`。
4. 不得以一般生理訊號分析慣例自行補充研究者未核准的方法。
5. 不得改變輸入資料的原始值。
6. 所有分析結果都必須能追溯到 Participant / Time / Condition / Block。
7. 需要修正規格中的矛盾或歧義時，不得自行選擇其中一個版本；必須先標記問題。

---

# 1. Analysis Scope

目前只實作：

- HRV Time-Domain Analysis
- Mean RR
- SDNN
- RMSSD

輸入資料為前處理完成後的 **NNi**。

本分析不負責 ECG preprocessing、R-peak detection、ectopic beat correction 等前處理。

---

# 2. Experimental Hierarchy

## 2.1 Block-level

Block-level 的定義：

- Pre-resting EC1
- Pre-resting EC2
- Pre-resting EC3
- Pre-resting EO1
- Pre-resting EO2
- Pre-resting EO3
- Post-resting EC1
- Post-resting EC2
- Post-resting EC3
- Post-resting EO1
- Post-resting EO2
- Post-resting EO3
- Task（若原始資料結構將 Task 視為單一 block，依資料中的定義處理）

其中：

- EC = Eyes Closed
- EO = Eyes Open

原始研究定義：

> Block-level: pre and post 的 EC 和 EO 個別計算，例如 pre-resting EC1、EC2、EC3。

## 2.2 Condition-level

Condition-level 包含：

1. Pre-resting EC
2. Pre-resting EO
3. Post-resting EC
4. Post-resting EO
5. Task

Condition-level 的計算方式：

> 將同一 condition 下的三個 block-level 數值相加後除以 3。

例如：

`Pre-resting EC = (EC1 + EC2 + EC3) / 3`

### 重要

Condition-level 不是直接重新把三個 block 的原始 NNi 串接後重新計算 HRV。

應使用：

`block-level metric → condition-level aggregation`

即：

`condition_metric = mean(block_metric_1, block_metric_2, block_metric_3)`

但如果資料缺少某一 block，是否改為使用實際存在的 block 數量計算平均，**目前規格未定義，不得自行決定**。

---

# 3. Required Analysis Levels

以下三個指標都必須同時進行：

## 3.1 Mean RR

必須計算：

- Block-level Mean RR
- Condition-level Mean RR

## 3.2 SDNN

必須計算：

- Block-level SDNN
- Condition-level SDNN

## 3.3 RMSSD

必須計算：

- Block-level RMSSD
- Condition-level RMSSD

---

# 4. Input

## 4.1 Accepted file formats

允許：

- `.xlsx`
- `.csv`

主要輸入資料為 NNi。

## 4.2 Required schema validation

讀取資料後，必須進行 schema validation。

至少需要能辨識：

- `Participant`
- `Time`
- `Condition`
- `Block`

以及 NNi 數值欄位。

### 注意

目前提供的研究規格沒有明確指定 NNi 欄位的實際欄名。

因此：

- 不得自行猜測欄位名稱。
- 如果目前 repository 已有明確 schema/config，應使用既有 schema。
- 如果沒有明確定義，標記 `REQUIRES_HUMAN_DECISION`。

---

# 5. Mean RR

Mean RR 的輸入為該分析單位內的 NNi。

## 5.1 Block-level

對每一個 block 分別計算：

`Mean RR = mean(NNi)`

例如：

- Pre EC1
- Pre EC2
- Pre EC3
- ...

不得跨 block 合併原始 NNi 後才計算。

## 5.2 Condition-level

先取得各 block-level Mean RR，再進行 condition-level aggregation。

例如：

`Pre EC Mean RR = mean(Pre EC1 Mean RR, Pre EC2 Mean RR, Pre EC3 Mean RR)`

---

# 6. SDNN

## 6.1 Block-level

對每一個 block 的 NNi 計算 SDNN。

目前研究規格只指定：

`SDNN`

但**未指定自由度 / estimator（例如 sample SD 或 population SD）**。

因此：

- 不得自行選擇。
- 應沿用專案既有、已核准的統計定義。
- 若專案中沒有既有定義，標記 `REQUIRES_HUMAN_DECISION`。

## 6.2 Condition-level

Condition-level SDNN 的研究規格為：

> 先取得 block-level 數值，再將三個 block-level 數值相加後除以三。

因此預設實作邏輯：

`Condition SDNN = mean(Block SDNN 1, Block SDNN 2, Block SDNN 3)`

**不要將三個 block 的 NNi 直接串接後重新計算 SDNN。**

---

# 7. RMSSD

## 7.1 Block-level

對每一個 block 的 NNi 計算 RMSSD。

目前研究規格沒有指定：

- 是否先進行額外 filtering
- 是否對差分值做額外處理
- 是否使用其他 correction

因此不得自行新增。

## 7.2 Condition-level

Condition-level RMSSD 使用 block-level 結果進行 aggregation：

`Condition RMSSD = mean(Block RMSSD 1, Block RMSSD 2, Block RMSSD 3)`

**不要將三個 block 的 NNi 直接串接後重新計算 RMSSD。**

---

# 8. Statistical Analysis

本版本的圖表需求**不包含顯著差異比較**。

因此本 time-domain visualization specification：

- 不需要進行顯著差異檢定
- 不需要在圖中標示 `*`
- 不需要在圖中標示 `**`
- 不需要為圖表產生 p-value annotation

目前若 repository 已存在其他研究統計分析模組，Codex 不得因本文件的圖表需求而自行新增或修改統計檢定。

除非另有獨立且已核准的 statistical specification，否則本文件只負責：

`descriptive summary + 95% CI visualization`


---

# 9. Confidence Interval

所有指定圖表均必須呈現：

`mean ± 95% CI`

目前研究要求是**視覺化 95% CI，不進行差異比較**。

但目前規格沒有明確指定：

- CI 的計算方法
- t-based CI 或其他方法
- subject-level aggregation 的具體計算方式

因此：

- 不得自行決定 CI estimator。
- 若專案已有統計設定，沿用既有設定。
- 否則標記 `REQUIRES_HUMAN_DECISION`。

圖表中的 CI 必須清楚可辨識。

---

# 10. Required Figures

# 10.1 Mean RR — Block-level

輸出一張 figure，figure 內包含兩個 panel：

- Left panel = Pre
- Right panel = Post

例如：

Pre-resting：
- EC1
- EC2
- EC3
- EO1
- EO2
- EO3

Post-resting：
- EC1
- EC2
- EC3
- EO1
- EO2
- EO3

圖形：

- line plot
- mean ± 95% CI

顯著差異：

- `p < .05` → `*`
- `p < .001` → `**`

---

# 10.2 Mean RR — Condition-level

輸出一張 figure。

Conditions：

- Pre-resting EC
- Pre-resting EO
- Post-resting EC
- Post-resting EO
- Task

圖形：

- line plot
- mean ± 95% CI

顯著差異：

- `p < .05` → `*`
- `p < .001` → `**`

---

# 10.3 SDNN — Block-level

輸出一張 figure，figure 內包含兩個 panel：

- Left panel = Pre
- Right panel = Post

例如：

Pre：
- EC1
- EC2
- EC3
- EO1
- EO2
- EO3

Post：
- EC1
- EC2
- EC3
- EO1
- EO2
- EO3

圖形類型：

- **不限制圖形類型**
- Codex 不得自行將「不限制」解讀為必須使用 histogram、line plot 或其他特定圖形
- 圖中必須呈現 mean ± 95% CI

不需要進行或標示顯著差異比較。

---

# 10.4 SDNN — Condition-level

輸出一張 figure。

Conditions：

- Pre-resting EC
- Pre-resting EO
- Post-resting EC
- Post-resting EO
- Task（若依資料定義包含 Task）

圖形類型：

- **不限制圖形類型**
- 圖中必須呈現 mean ± 95% CI

不需要進行或標示顯著差異比較。

---

# 10.5 RMSSD — Block-level

輸出一張 figure，figure 內包含兩個 panel：

- Left panel = Pre
- Right panel = Post

圖形類型：

- **不限制圖形類型**
- 圖中必須呈現 mean ± 95% CI

不需要進行或標示顯著差異比較。

---

# 10.6 RMSSD — Condition-level

輸出一張 figure。

Conditions：

- Pre-resting EC
- Pre-resting EO
- Post-resting EC
- Post-resting EO
- Task

圖形類型：

- **不限制圖形類型**
- 圖中必須呈現 mean ± 95% CI

不需要進行或標示顯著差異比較。

---

# 12. Color / Style

目前不限制 SDNN / RMSSD 的圖形類型。

因此：

- 不得將藍線、特定線型等舊規格強制套用到所有圖形。
- CI 必須與主要資料呈現方式清楚區分。
- 不得自行新增研究者未指定的視覺編碼。
- 不得用 seaborn 等方式改變整體視覺風格，除非 repository 已有統一 plotting specification。

---

# 13. Output Data

建議至少輸出能追溯以下資訊的 analysis table：

- Participant
- Time
- Condition
- Block
- Analysis level
- Mean RR
- SDNN
- RMSSD

Condition-level 必須能追溯到其使用的 block-level values。

不要只輸出最後的 group mean。

---

# 14. Traceability

每一個結果都應可以追溯：

`Raw NNi`
→ `validated NNi`
→ `Block-level metric`
→ `Condition-level metric`
→ `Group summary`
→ `Statistical result`
→ `Figure`

如果某一步產生 exclusion / missing data，必須留下可追溯資訊。

---

# 15. Validation Requirements

實作完成後必須檢查：

## Input validation

- [ ] xlsx 可以讀取
- [ ] csv 可以讀取
- [ ] schema validation 正常
- [ ] Participant 欄位存在
- [ ] Time 欄位存在
- [ ] Condition 欄位存在
- [ ] Block 欄位存在
- [ ] NNi 欄位存在或使用已核准 schema

## Analysis validation

- [ ] Mean RR block-level
- [ ] Mean RR condition-level
- [ ] SDNN block-level
- [ ] SDNN condition-level
- [ ] RMSSD block-level
- [ ] RMSSD condition-level

## Aggregation validation

- [ ] Condition-level 是由 block-level metric aggregation 而來
- [ ] 不得直接串接三個 block 的 NNi 重新計算 condition-level SDNN/RMSSD
- [ ] Pre / Post 不得錯置
- [ ] EC / EO 不得錯置

## Figure validation

- [ ] Mean RR block-level figure
- [ ] Mean RR condition-level figure
- [ ] SDNN block-level figure
- [ ] SDNN condition-level figure
- [ ] RMSSD block-level figure
- [ ] RMSSD condition-level figure

---

# 16. Do Not Implement Without Human Approval

以下項目目前不得由 Codex 自行決定：

1. SDNN 的 SD estimator / degrees of freedom
2. CI 的計算方法
3. statistical test（本文件圖表目前不需要差異檢定）
4. multiple-comparison correction
5. effect size
6. missing block 的 condition-level aggregation
7. 缺失 NNi 的處理
8. outlier handling
9. additional NNi filtering
10. SDNN / RMSSD 的具體圖形類型
11. 未在本 specification 中定義的任何 preprocessing
12. 未在本 specification 中定義的任何 statistical modeling

---

# 17. Implementation Principle

Codex 的角色是：

**「依照已核准 specification 實作」**

而不是：

**「根據一般 HRV 知識替研究者決定方法」**

如果發現：

- 規格矛盾
- 欄位不明
- 統計方法未定義
- 圖表規格不一致
- 資料結構不足

應停止該決策點並回報：

`REQUIRES_HUMAN_DECISION`

不要自行補上看似合理的方法。

---

# 18. Source of This Specification

本 specification 整理自研究者提供的 HRV time-domain analysis 原始規格。

原始規格明確定義：

- Block-level / Condition-level
- Mean RR
- SDNN
- RMSSD
- xlsx / csv input
- schema validation
- required figures
- mean ± 95% CI
- significance annotation thresholds

其中未明確定義的研究方法，刻意保留為待人工核准事項。
