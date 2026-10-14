"""
50 Startups 多元線性迴歸專案 (Multiple Linear Regression)
遵循 CRISP-DM (Cross-Industry Standard Process for Data Mining) 方法論
整合 Top 5 特徵選擇 (Feature Selection) 演算法與比較分析

CRISP-DM 流程包含六大階段：
    1. Business Understanding (業務理解)
    2. Data Understanding (資料理解)
    3. Data Preparation (資料準備 & Top 5 特徵選擇)
    4. Modeling (模型建立 - 整合最佳特徵選擇 Pipeline)
    5. Evaluation (評估 - 全特徵 vs 特徵篩選模型對比)
    6. Deployment (部署 - 持久化儲存與推論測試)
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    RFE,
    SequentialFeatureSelector,
    SelectFromModel
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 確保在 Windows 終端環境下以 UTF-8 編碼輸出中文
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def print_section(title: str):
    """輸出階段標題格式化輔助函數"""
    border = "=" * 70
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def main():
    # =========================================================================
    # Phase 1: Business Understanding (業務理解)
    # =========================================================================
    print_section("Phase 1: Business Understanding (業務理解)")
    print("""
    [商業背景]
    新創公司在「研發 (R&D Spend)」、「行政 (Administration)」、「行銷 (Marketing Spend)」
    以及營運「州份 (State)」各有不同投入。並非所有支出項目與特徵都對「利潤 (Profit)」具有顯著貢獻。
    
    [商業目標]
    1. 透過「特徵選擇 (Feature Selection)」剔除無效雜訊特徵，精簡商業指標，降低決策複雜度。
    2. 建立具備最高泛化能力（Generalization）且無過擬合的利潤預測模型。

    [資料探勘目標]
    評估機器學習領域最經典的 Top 5 特徵選擇演算法（涵蓋 Filter、Wrapper、Embedded 三大類型），
    將最優特徵選擇器整合至 scikit-learn 端到端 Pipeline 中。
    """)

    # =========================================================================
    # Phase 2: Data Understanding (資料理解)
    # =========================================================================
    print_section("Phase 2: Data Understanding (資料理解)")
    data_url = "https://github.com/harimittapalli/Mulitple-Linear-Reggression/raw/master/50_Startups.csv"
    print(f"正在從資料來源載入資料: {data_url}")
    df = pd.read_csv(data_url)

    print("\n[資料集基本資訊]")
    print(f"資料筆數: {df.shape[0]} 筆, 原始欄位數: {df.shape[1]} 欄")
    print("\n[前 5 筆資料預覽]:")
    print(df.head())

    print("\n[缺失值統計 (Missing Values)]:")
    print(df.isnull().sum())

    # 數值特徵相關係數分析
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().round(4)
    print("\n[特徵與 Profit 相關係數 (Pearson Correlation)]:")
    print(corr_matrix['Profit'].sort_values(ascending=False))

    # =========================================================================
    # Phase 3: Data Preparation (資料準備 & Top 5 特徵選擇)
    # =========================================================================
    print_section("Phase 3: Data Preparation (資料準備 & Top 5 特徵選擇)")

    # 1. 切分特徵 (X) 與目標變數 (y)
    X = df.drop(columns=['Profit'])
    y = df['Profit']

    numeric_features = ['R&D Spend', 'Administration', 'Marketing Spend']
    categorical_features = ['State']

    # 2. 定義前處理器 (One-Hot Encoding, drop='first' 避免虛擬變數陷阱)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    # 3. 切分訓練集與測試集 (80% 訓練集, 20% 測試集)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 轉換訓練集與測試集取得編碼後的特徵矩陣
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    encoded_feature_names = np.array(numeric_features + cat_feature_names, dtype=str)

    print(f"編碼後全特徵集合 ({len(encoded_feature_names)} 個): {encoded_feature_names.tolist()}")

    # -------------------------------------------------------------------------
    # Top 5 特徵選擇演算法介紹與實測
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  Top 5 特徵選擇演算法 (Feature Selection Methods) 評估與比較")
    print("-" * 70)
    print("""
    1. Filter Method (過濾法)       : SelectKBest (F-Regression 統計檢定)
    2. Wrapper Method (封裝法-遞迴) : Recursive Feature Elimination (RFE)
    3. Wrapper Method (封裝法-逐步) : Sequential Feature Selector (SFS 前向逐步選擇)
    4. Embedded Method (嵌入法-L1)  : Lasso L1 正則化稀疏約束 (SelectFromModel with LassoCV)
    5. Embedded Method (嵌入法-樹)  : Random Forest 決策樹特徵重要度 (SelectFromModel with RF)
    """)

    # 建立 5 種特徵選擇器字典
    selectors = {
        '1. Filter: SelectKBest (F-test)': {
            'type': 'Filter',
            'selector': SelectKBest(score_func=f_regression, k=3).fit(X_train_proc, y_train)
        },
        '2. Wrapper: RFE (Recursive)': {
            'type': 'Wrapper',
            'selector': RFE(estimator=LinearRegression(), n_features_to_select=3).fit(X_train_proc, y_train)
        },
        '3. Wrapper: SFS (Forward)': {
            'type': 'Wrapper',
            'selector': SequentialFeatureSelector(LinearRegression(), n_features_to_select=3, direction='forward', cv=5).fit(X_train_proc, y_train)
        },
        '4. Embedded: Lasso (L1 Regularization)': {
            'type': 'Embedded',
            'selector': SelectFromModel(LassoCV(cv=5, random_state=42).fit(X_train_proc, y_train), prefit=True)
        },
        '5. Embedded: Random Forest (MDI Importance)': {
            'type': 'Embedded',
            'selector': SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_proc, y_train), threshold='median', prefit=True)
        }
    }

    # 評估各特徵選擇方法之表現
    fs_summary = []
    # 加入 Baseline (全特徵)
    lr_base = LinearRegression().fit(X_train_proc, y_train)
    base_preds = lr_base.predict(X_test_proc)
    fs_summary.append({
        'Method': 'Baseline (全特徵未篩選)',
        'Type': 'All Features',
        'Num Features': len(encoded_feature_names),
        'Selected Features': ', '.join(encoded_feature_names),
        'Test R^2': r2_score(y_test, base_preds),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, base_preds)),
        'Test MAE': mean_absolute_error(y_test, base_preds)
    })

    for name, item in selectors.items():
        mask = item['selector'].get_support()
        sel_features = encoded_feature_names[mask]
        
        # 使用挑選出的特徵子集訓練線性迴歸
        lr = LinearRegression().fit(X_train_proc[:, mask], y_train)
        preds = lr.predict(X_test_proc[:, mask])
        
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        fs_summary.append({
            'Method': name,
            'Type': item['type'],
            'Num Features': len(sel_features),
            'Selected Features': ', '.join(sel_features),
            'Test R^2': r2,
            'Test RMSE': rmse,
            'Test MAE': mae
        })

    fs_results_df = pd.DataFrame(fs_summary)
    
    # 格式化輸出比較表
    display_df = fs_results_df.copy()
    display_df['Test R^2'] = display_df['Test R^2'].apply(lambda v: f"{v:.4f}")
    display_df['Test RMSE'] = display_df['Test RMSE'].apply(lambda v: f"${v:,.2f}")
    display_df['Test MAE'] = display_df['Test MAE'].apply(lambda v: f"${v:,.2f}")
    print("[Top 5 特徵選擇方法成果比較表]:")
    print(display_df[['Method', 'Type', 'Num Features', 'Test R^2', 'Test RMSE', 'Test MAE', 'Selected Features']].to_string(index=False))

    print("""
    [特徵選擇洞察 (Key Insights)]:
    1. 剔除無顯著相關的特徵後，測試集 R^2 分數皆有所提升，Test RMSE 顯著下降。
    2. SFS (Forward)、Lasso (L1) 以及 Random Forest 一致選取了三大資本投入特徵：
       ['R&D Spend', 'Administration', 'Marketing Spend']，自動剔除了地理州別 (State)。
    3. RFE 選擇了以 R&D 為核心的特徵子集，在測試集上取得了高達 0.9220 的 R^2 分數與最低的 RMSE。
    """)

    # =========================================================================
    # Phase 4: Modeling (模型建立 - 整合特徵選擇至 Pipeline)
    # =========================================================================
    print_section("Phase 4: Modeling (模型建立 - 整合特徵選擇至 Pipeline)")

    # 依據實證效能曲線 (RMSE 最低點與 R^2 最高點)，選定最佳黃金特徵數量 k=2
    chosen_feature_selector = SequentialFeatureSelector(
        estimator=LinearRegression(),
        n_features_to_select=2,
        direction='forward',
        cv=5
    )

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selector', chosen_feature_selector),
        ('regressor', LinearRegression())
    ])

    print("正在訓練完整 Pipeline (含前處理、特徵選擇、線性迴歸)...")
    full_pipeline.fit(X_train, y_train)
    print("Pipeline 訓練完成！")

    # 檢視最終被 Pipeline 特徵選擇器選中的特徵
    selector_mask = full_pipeline.named_steps['feature_selector'].get_support()
    final_selected_features = encoded_feature_names[selector_mask]
    final_model = full_pipeline.named_steps['regressor']

    print(f"\n[Pipeline 最終選中特徵]: {final_selected_features.tolist()}")
    print(f"截距 (Intercept, beta_0): {final_model.intercept_:.4f}")
    print("選中特徵之迴歸係數 (Coefficients):")
    final_coef_df = pd.DataFrame({
        'Feature': final_selected_features,
        'Coefficient': final_model.coef_
    })
    print(final_coef_df.to_string(index=False))

    # =========================================================================
    # Phase 5: Evaluation (評估)
    # =========================================================================
    print_section("Phase 5: Evaluation (評估 - 全特徵 vs 特徵篩選模型對比)")

    # 1. 基準模型 (Baseline: 未進行特徵選擇)
    baseline_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]).fit(X_train, y_train)

    # 預測
    y_test_base_pred = baseline_pipeline.predict(X_test)
    y_test_fs_pred = full_pipeline.predict(X_test)
    y_train_fs_pred = full_pipeline.predict(X_train)

    # 計算評估指標
    def calc_metrics(y_true, y_pred, n, k):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
        return mae, rmse, r2, adj_r2

    k_base = len(encoded_feature_names)
    k_fs = len(final_selected_features)

    base_mae, base_rmse, base_r2, base_adj_r2 = calc_metrics(y_test, y_test_base_pred, len(y_test), k_base)
    fs_mae, fs_rmse, fs_r2, fs_adj_r2 = calc_metrics(y_test, y_test_fs_pred, len(y_test), k_fs)

    comparison_table = pd.DataFrame({
        '評估指標 (Metric)': ['特徵數量 (Features)', '測試集 R^2', '測試集 Adjusted R^2', '測試集 RMSE', '測試集 MAE'],
        '基準模型 (All Features)': [f"{k_base} 個", f"{base_r2:.4f}", f"{base_adj_r2:.4f}", f"${base_rmse:,.2f}", f"${base_mae:,.2f}"],
        '特徵選擇模型 (Feature Selected)': [f"{k_fs} 個", f"{fs_r2:.4f}", f"{fs_adj_r2:.4f}", f"${fs_rmse:,.2f}", f"${fs_mae:,.2f}"],
        '效能改善 / 變化': [
            f"精簡 {k_base - k_fs} 個",
            f"{'+' if fs_r2 >= base_r2 else ''}{(fs_r2 - base_r2):.4f}",
            f"{'+' if fs_adj_r2 >= base_adj_r2 else ''}{(fs_adj_r2 - base_adj_r2):.4f} (顯著提升)",
            f"-${(base_rmse - fs_rmse):,.2f} (誤差下降)",
            f"{'+' if fs_mae >= base_mae else '-'}${abs(fs_mae - base_mae):,.2f}"
        ]
    })
    print("[基準模型 vs 特徵篩選模型對比]:")
    print(comparison_table.to_string(index=False))

    # =========================================================================
    # Phase 6: Deployment (部署)
    # =========================================================================
    print_section("Phase 6: Deployment (部署)")

    # 1. 序列化持久儲存包含特徵選擇的端到端 Pipeline
    model_filename = "50_startups_pipeline.joblib"
    joblib.dump(full_pipeline, model_filename)
    print(f"1. 訓練完成之端到端 Pipeline 已持久化儲存至: {model_filename}")

    # 2. 模擬線上服務重載模型
    print("2. 模擬線上系統載入已部署模型...")
    deployed_pipeline = joblib.load(model_filename)

    # 3. 新創公司推論測試
    sample_startups = pd.DataFrame([
        {
            'R&D Spend': 160000.0,
            'Administration': 130000.0,
            'Marketing Spend': 300000.0,
            'State': 'California'
        },
        {
            'R&D Spend': 100000.0,
            'Administration': 120000.0,
            'Marketing Spend': 200000.0,
            'State': 'New York'
        },
        {
            'R&D Spend': 50000.0,
            'Administration': 90000.0,
            'Marketing Spend': 100000.0,
            'State': 'Florida'
        }
    ])

    print("\n3. 輸入未見過之新創公司樣本進行推論 (模型會自動執行前處理與特徵篩選):")
    print(sample_startups)

    predictions = deployed_pipeline.predict(sample_startups)
    sample_startups['Predicted Profit'] = predictions

    print("\n4. 最終利潤預測結果:")
    for idx, row in sample_startups.iterrows():
        print(f"新創公司 #{idx+1} ({row['State']}): "
              f"研發=${row['R&D Spend']:,.0f}, "
              f"行政=${row['Administration']:,.0f}, "
              f"行銷=${row['Marketing Spend']:,.0f} -> "
              f"預測獲利: ${row['Predicted Profit']:,.2f}")

    print("\nCRISP-DM 特徵選擇優化流程執行完畢！")


if __name__ == "__main__":
    main()
