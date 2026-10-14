# 多元線性迴歸與特徵選擇雙資料集互動分析專案 (CRISP-DM)

本專案遵循國際標準資料探勘方法論 **CRISP-DM (Cross-Industry Standard Process for Data Mining)**，使用 Python 與 **Scikit-Learn** 針對兩大經典迴歸資料集：
1. **50 Startups** (50 家新創獲利預測, `50_Startups.csv`)
2. **Boston Housing** (波士頓房價中位數預測, `BostonHousing.csv`)

實作全套前處理、建模評估與 **Top 5 特徵選擇 (Feature Selection)** 演算法（涵蓋 Filter、Wrapper 與 Embedded 三大流派），實證找出**特徵數最佳甜點區 (Sweet Spot)**，並建立現代化**響應式雙資料集切換互動視覺化儀表板 (Interactive Web Dashboard)**。

---

## 🌐 互動展示網頁 (Interactive Demo)

本專案附帶可獨立執行的互動式網頁應用程式，具備 **雙資料集即時切換下拉式選單**、**Chart.js 動態走勢圖**、**Top 5 演算法多線切換比對** 與 **即時推論計算機 (Inference Calculator)**。

* **本地即時預覽**：直接雙擊開啟工作目錄中的 [`index.html`](./index.html)。
* **本地 HTTP 伺服器啟動方式**（推薦）：
  ```powershell
  python -m http.server 8000
  ```
  啟動後於瀏覽器訪問：[http://localhost:8000/index.html](http://localhost:8000/index.html)

---

## 📊 雙資料集核心實驗結論與甜點區對比

| 資料集 | 樣本數 | 候選特徵數 | 最佳甜點區 ($k$) | 選取核心特徵 | 測試集 $R^2$ | 測試集 RMSE | 改善幅度說明 |
|---|:---:|:---:|:---:|---|:---:|:---:|---|
| **50 Startups** | 50 筆 | 5 個 | **$k=2$ ⭐** | `['R&D Spend', 'Marketing Spend']` | **0.9474** | **$8,198.80** | 剔除行政支出與州別雜訊，誤差下降 $939.19 |
| **Boston Housing** | 506 筆 | 13 個 | **$k=7$ ⭐** | `['nox', 'rm', 'dis', 'rad', 'tax', 'ptratio', 'lstat']` | **0.6727** | **$4,899.54** | 特徵精簡 46.2%，泛化誤差超越 13 全特徵基準 ($4,928.60) |

---

## 🔬 Top 5 特徵選擇演算法對照

本專案實測並比較了以下 5 大經典演算法的逐步特徵選擇進程 (k=1..N)：
1. **Forward SFS (Sequential Feature Selection)**: 循序前向逐步選擇，每回合透過 5 折交叉驗證最大化評估分數。
2. **RFE (Recursive Feature Elimination)**: 遞迴特徵消除法，迭代剔除權重絕對值最小的特徵。
3. **Lasso L1 Regularization**: 透過 L1 稀疏約束懲罰項，自動將無顯著關聯或共線性的特徵係數直接壓縮歸零。
4. **ElasticNet**: 兼顧 L1 與 L2 正則化，穩定處理群組共線性特徵。
5. **Exhaustive Search / Best Subset Selection**: 窮舉搜尋各特徵數量下表現最佳的特徵子集。

---

## 📂 專案檔案結構 (Project Structure)

對齊課堂標準結構，完整包含所有進程 CSV、視覺化圖表與端到端模型：

```text
DIC_Linear_regression/
├── main.py                                  # 主執行入口 (可一鍵執行雙資料集完整分析)
├── 50_startups_crisp_dm.py                  # 50 Startups CRISP-DM 六大階段完整程式
├── boston_housing_crisp_dm.py               # Boston Housing CRISP-DM 六大階段完整程式
├── feature_progression_analysis.py          # 5 大演算法特徵進程計算、CSV 產出與繪圖腳本
│
├── 50_startups_pipeline.joblib              # 50 Startups 端到端持久化模型 (k=2)
├── boston_housing_pipeline.joblib           # Boston Housing 端到端持久化模型 (k=7)
│
├── forward_sfs_progression.csv              # Forward SFS 特徵逐步進程表
├── forward_sfs_plot.png                     # Forward SFS 特徵曲線圖
├── rfe_progression.csv                      # RFE 特徵逐步進程表
├── rfe_plot.png                             # RFE 特徵曲線圖
├── lasso_l1_ranking_progression.csv         # Lasso 特徵逐步進程表
├── lasso_l1_ranking_plot.png                # Lasso 特徵曲線圖
├── elasticnet_ranking_progression.csv       # ElasticNet 特徵逐步進程表
├── elasticnet_ranking_plot.png              # ElasticNet 特徵曲線圖
├── exhaustive_search_progression.csv        # 窮舉最佳子集特徵逐步進程表
├── exhaustive_search_plot.png               # 窮舉最佳子集曲線圖
│
├── feature_selection_comparison.csv         # 5 大演算法成果整合比對表
├── feature_selection_comparison.png         # 5 大演算法對比圖表
├── all_5_algorithms_comparison_plot.png     # 5 大演算法全覽對比圖 (RMSE & R²)
├── boston_housing_feature_selection_curves.png # 波士頓房價 5 大演算法曲線對比圖
│
├── model_data.json                          # 前端資料集核心數據 (JSON 格式)
├── model_data.js                            # 前端資料集核心數據 (JS 腳本格式，防 CORS)
├── index.html                               # 現代化響應式雙資料集切換互動儀表板
├── requirements.txt                         # 專案相依套件清單
├── .gitignore                               # Git 忽略設定
└── README.md                                # 專案說明文件
```

---

## 🚀 快速執行說明 (How to Run)

### 1. 安裝環境相依套件
```powershell
pip install -r requirements.txt
```

### 2. 一鍵執行完整管線 (包含特徵進程分析與模型儲存)
```powershell
python main.py
```
或個別執行：
```powershell
# 執行 50 Startups 流程
python 50_startups_crisp_dm.py

# 執行 Boston Housing 流程
python boston_housing_crisp_dm.py

# 產生所有演算法逐步 CSV 與視覺化圖表
python feature_progression_analysis.py
```

### 3. 開啟互動式網頁儀表板
```powershell
python -m http.server 8000
```
瀏覽器訪問 `http://localhost:8000/index.html`，即可透過頂部下拉式選單在 **50 Startups** 與 **Boston Housing** 之間切換。
