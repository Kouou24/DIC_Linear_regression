"""
50 Startups 特徵數量與模型效能分析 (RMSE & R-squared by Number of Features)
評估 Top 5 特徵選擇方法在不同特徵數量 (k=1..5) 下的變化曲線，並繪製對比圖
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 確保 UTF-8 輸出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 70)
    print("  50 Startups: 特徵選擇 (k=1..5) 效能分析與曲線繪製")
    print("=" * 70)

    # 1. 載入資料
    data_url = "https://github.com/harimittapalli/Mulitple-Linear-Reggression/raw/master/50_Startups.csv"
    df = pd.read_csv(data_url)
    df_ohe = pd.get_dummies(df, columns=['State'], drop_first=True, dtype=float)
    features = np.array([c for c in df_ohe.columns if c != 'Profit'])

    # 採用與使用者相同的劃分配置以精確重現結果
    train, test = train_test_split(df_ohe, test_size=0.2, random_state=0)
    X_train, y_train = train[features], train['Profit']
    X_test, y_test = test[features], test['Profit']

    # 2. 準備 5 種特徵選擇器在 k=1..5 的選取特徵
    # (1) Filter: SelectKBest (F-test)
    skb_scores = f_regression(X_train, y_train)[0]
    skb_ranked = features[np.argsort(skb_scores)[::-1]]

    # (2) Wrapper: SFS (Forward)
    sfs_results = {}
    for k in range(1, 5):
        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction='forward', cv=5).fit(X_train, y_train)
        sfs_results[k] = list(features[sfs.get_support()])
    sfs_results[5] = list(features)

    # (3) Wrapper: RFE (搭配標準化避免尺度偏差)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    rfe_selector = RFE(LinearRegression(), n_features_to_select=1).fit(X_train_scaled, y_train)
    rfe_ranked = features[np.argsort(rfe_selector.ranking_)]

    # (4) Embedded: Lasso L1
    lasso = LassoCV(cv=5, random_state=0).fit(X_train, y_train)
    lasso_ranked = features[np.argsort(np.abs(lasso.coef_))[::-1]]

    # (5) Embedded: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train, y_train)
    rf_ranked = features[np.argsort(rf.feature_importances_)[::-1]]

    # (6) 使用者圖片中的原始逐步特徵組合 (User Reference)
    user_ref_features = {
        1: ['R&D Spend'],
        2: ['R&D Spend', 'Marketing Spend'],
        3: ['R&D Spend', 'Marketing Spend', 'State_New York'],
        4: ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida'],
        5: ['R&D Spend', 'Marketing Spend', 'Administration', 'State_Florida', 'State_New York']
    }

    all_methods = {
        '使用者參考基準 (User Reference)': user_ref_features,
        '1. Filter (SelectKBest)': {k: list(skb_ranked[:k]) for k in range(1, 6)},
        '2. Wrapper (SFS Forward)': sfs_results,
        '3. Wrapper (RFE Scaled)': {k: list(rfe_ranked[:k]) for k in range(1, 6)},
        '4. Embedded (Lasso L1)': {k: list(lasso_ranked[:k]) for k in range(1, 6)},
        '5. Embedded (Random Forest)': {k: list(rf_ranked[:k]) for k in range(1, 6)},
    }

    results_data = []

    # 計算各方法各 k 值的指標
    for method_name, k_dict in all_methods.items():
        print(f"\n--- {method_name} ---")
        for k in range(1, 6):
            sel_feats = k_dict[k]
            lr = LinearRegression().fit(X_train[sel_feats], y_train)
            preds = lr.predict(X_test[sel_feats])
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            results_data.append({
                'Method': method_name,
                'Number of Features': k,
                'Selected Features': str(sel_feats),
                'RMSE': rmse,
                'R-squared': r2
            })
            print(f"k={k}: {sel_feats} -> RMSE: {rmse:.6f}, R^2: {r2:.6f}")

    results_df = pd.DataFrame(results_data)

    # 3. 輸出使用者參考基準的完整表格 (同圖片 2)
    user_table = results_df[results_df['Method'] == '使用者參考基準 (User Reference)'].copy()
    print("\n" + "=" * 70)
    print("  重現使用者圖片 2 的表格 (Exact Reproduction Table):")
    print("=" * 70)
    print(user_table[['Number of Features', 'Selected Features', 'RMSE', 'R-squared']].to_string(index=False))

    # 4. 繪製圖表 (同圖片 1 的風格)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 繪製使用者基準曲線 (加粗標註)
    user_rows = results_df[results_df['Method'] == '使用者參考基準 (User Reference)']
    ax1.plot(user_rows['Number of Features'], user_rows['RMSE'], marker='o', linewidth=2.5, color='#1f77b4', label='User Reference (Sweet Spot k=2)')
    ax2.plot(user_rows['Number of Features'], user_rows['R-squared'], marker='o', linewidth=2.5, color='#1f77b4', label='User Reference (Sweet Spot k=2)')

    # 繪製其餘 5 種演算法曲線做對比
    styles = {
        '1. Filter (SelectKBest)': ('#ff7f0e', '--', 's'),
        '2. Wrapper (SFS Forward)': ('#2ca02c', '-.', '^'),
        '3. Wrapper (RFE Scaled)': ('#d62728', ':', 'v'),
        '4. Embedded (Lasso L1)': ('#9467bd', '--', 'D'),
        '5. Embedded (Random Forest)': ('#8c564b', ':', 'x')
    }

    for method, (color, linestyle, marker) in styles.items():
        m_rows = results_df[results_df['Method'] == method]
        ax1.plot(m_rows['Number of Features'], m_rows['RMSE'], marker=marker, linestyle=linestyle, alpha=0.8, color=color, label=method)
        ax2.plot(m_rows['Number of Features'], m_rows['R-squared'], marker=marker, linestyle=linestyle, alpha=0.8, color=color, label=method)

    # 設定子圖 1: RMSE
    ax1.set_title('RMSE by Number of Features', fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel('Number of Features', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 設定子圖 2: R-squared
    ax2.set_title('R-squared by Number of Features', fontsize=14, fontweight='bold', pad=12)
    ax2.set_xlabel('Number of Features', fontsize=12)
    ax2.set_ylabel('R-squared', fontsize=12)
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.legend(fontsize=9, loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_img = "feature_selection_curves.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n[成功生成視覺化圖表]: 已儲存至 {output_img}")


if __name__ == "__main__":
    main()

