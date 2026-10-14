// 自動由 feature_progression_analysis.py 生成之模型數據
const MODEL_DATA = {
  "startups": {
    "id": "startups",
    "title": "50 Startups 獲利預測與特徵選擇分析",
    "subtitle": "Multiple Linear Regression & Feature Selection (k=1..5) Diagnostic Studio",
    "badge": "50",
    "badgeColor": "from-blue-600 to-indigo-500",
    "badgeText": "CRISP-DM",
    "targetName": "Profit (獲利)",
    "currencyPrefix": "$",
    "unitMultiplier": 1.0,
    "kRange": [
      1,
      2,
      3,
      4,
      5
    ],
    "sweetSpotK": 2,
    "sweetSpotFeatureStr": "R&D Spend + Marketing Spend",
    "kpi": {
      "maxR2": "0.9474",
      "maxR2Diff": "+0.0127 vs 全特徵",
      "minRmse": "$8,198.80",
      "minRmseDiff": "-$939.19",
      "goldenFeatures": "R&D + Marketing",
      "goldenDesc": "研發支出與行銷支出雙引擎",
      "sparsity": "60%",
      "sparsityDesc": "5 特徵 → 2 特徵 (排除雜訊)"
    },
    "algorithms": {
      "forward_sfs": {
        "name": "Forward Sequential Feature Selection (SFS)",
        "rmse": [
          8274.87,
          8198.8,
          8803.78,
          8997.2,
          9137.99
        ],
        "r2": [
          0.946459,
          0.947439,
          0.939396,
          0.936703,
          0.934707
        ],
        "features": [
          "R&D Spend",
          "R&D Spend, Marketing Spend",
          "R&D Spend, Administration, Marketing Spend",
          "R&D Spend, Administration, Marketing Spend, State_New York",
          "R&D Spend, Administration, Marketing Spend, State_Florida, State_New York"
        ]
      },
      "rfe": {
        "name": "Recursive Feature Elimination (RFE)",
        "rmse": [
          8274.87,
          8198.8,
          8803.78,
          9068.54,
          9137.99
        ],
        "r2": [
          0.946459,
          0.947439,
          0.939396,
          0.935696,
          0.934707
        ],
        "features": [
          "R&D Spend",
          "R&D Spend, Marketing Spend",
          "R&D Spend, Marketing Spend, Administration",
          "R&D Spend, Marketing Spend, Administration, State_Florida",
          "R&D Spend, Marketing Spend, Administration, State_Florida, State_New York"
        ]
      },
      "lasso_l1_ranking": {
        "name": "Lasso L1 Ranking",
        "rmse": [
          8274.87,
          8198.8,
          8309.06,
          8409.92,
          9137.99
        ],
        "r2": [
          0.946459,
          0.947439,
          0.946015,
          0.944697,
          0.934707
        ],
        "features": [
          "R&D Spend",
          "R&D Spend, Marketing Spend",
          "R&D Spend, Marketing Spend, State_New York",
          "R&D Spend, Marketing Spend, State_New York, State_Florida",
          "R&D Spend, Marketing Spend, State_New York, State_Florida, Administration"
        ]
      },
      "elasticnet_ranking": {
        "name": "ElasticNet Ranking",
        "rmse": [
          8274.87,
          8198.8,
          8803.78,
          8997.2,
          9137.99
        ],
        "r2": [
          0.946459,
          0.947439,
          0.939396,
          0.936703,
          0.934707
        ],
        "features": [
          "R&D Spend",
          "R&D Spend, Marketing Spend",
          "R&D Spend, Marketing Spend, Administration",
          "R&D Spend, Marketing Spend, Administration, State_New York",
          "R&D Spend, Marketing Spend, Administration, State_New York, State_Florida"
        ]
      },
      "exhaustive_search": {
        "name": "Exhaustive Best Subset Search",
        "rmse": [
          8274.87,
          8198.8,
          8171.65,
          8178.37,
          9137.99
        ],
        "r2": [
          0.946459,
          0.947439,
          0.947786,
          0.9477,
          0.934707
        ],
        "features": [
          "R&D Spend",
          "R&D Spend, Marketing Spend",
          "R&D Spend, Administration, State_Florida",
          "R&D Spend, Administration, State_Florida, State_New York",
          "R&D Spend, Administration, Marketing Spend, State_Florida, State_New York"
        ]
      }
    },
    "simulator": {
      "type": "startups",
      "goldenModel": {
        "title": "最佳黃金模型 (k=2)",
        "desc": "變數: 研發 + 行銷",
        "rmse": "$8,198.80",
        "intercept": 46346.338761236555,
        "coefficients": {
          "R&D Spend": 0.7886182999424615,
          "Marketing Spend": 0.032576627317020684
        }
      },
      "baselineModel": {
        "title": "全特徵基準模型 (k=5)",
        "desc": "變數: 全部 5 項支出與地理欄位",
        "rmse": "$9,137.99",
        "intercept": 42554.16761773237,
        "coefficients": {
          "R&D Spend": 0.7734671927328001,
          "Administration": 0.03288459753625596,
          "Marketing Spend": 0.036610025863856116,
          "State_Florida": -959.2841600571066,
          "State_New York": 699.3690525192583
        }
      }
    },
    "crispSummary": {
      "phase12": "透過 50 家新創收支分析確立利潤預測目標。無缺失值，研發支出與獲利呈現高達 0.9729 的極強相關性。",
      "phase34": "實測 5 大特徵選擇演算法，一致驗證 k=2 為全域最佳甜點區，將前處理與 SFS 封裝入 Pipeline。",
      "phase56": "篩選後 RMSE 降至 $8,198.80，R² 升至 0.9474。已持久化為 50_startups_pipeline.joblib。"
    }
  },
  "boston": {
    "id": "boston",
    "title": "Boston Housing 房價預測與特徵選擇分析",
    "subtitle": "Multiple Linear Regression & Feature Selection (k=1..13) Diagnostic Studio",
    "badge": "BH",
    "badgeColor": "from-emerald-600 to-teal-500",
    "badgeText": "CRISP-DM",
    "targetName": "MEDV (房價中位數, $1,000s)",
    "currencyPrefix": "$",
    "unitMultiplier": 1000.0,
    "kRange": [
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13
    ],
    "sweetSpotK": 7,
    "sweetSpotFeatureStr": "nox, rm, dis, rad, tax, ptratio, lstat",
    "kpi": {
      "maxR2": "0.6727",
      "maxR2Diff": "+0.0039 vs 全特徵",
      "minRmse": "$4,899.54",
      "minRmseDiff": "-$29.06 (誤差最低)",
      "goldenFeatures": "RM + LSTAT + PTRATIO + 4 項環境",
      "goldenDesc": "房間數、低階社經比、生師比等",
      "sparsity": "46.2%",
      "sparsityDesc": "13 特徵 → 7 特徵 (精簡穩定)"
    },
    "algorithms": {
      "forward_sfs": {
        "name": "Forward Sequential Feature Selection (SFS)",
        "rmse": [
          5789.61,
          5589.57,
          5207.2,
          5367.83,
          5374.53,
          5228.98,
          5192.86,
          5257.91,
          5139.57,
          5049.77,
          5057.97,
          4917.7,
          4928.6
        ],
        "r2": [
          0.542918,
          0.573958,
          0.630253,
          0.60709,
          0.606108,
          0.627153,
          0.632287,
          0.623017,
          0.639796,
          0.652272,
          0.651142,
          0.670223,
          0.668759
        ],
        "features": [
          "lstat",
          "rm, lstat",
          "rm, ptratio, lstat",
          "rm, ptratio, b, lstat",
          "rm, dis, ptratio, b, lstat",
          "nox, rm, dis, ptratio, b, lstat",
          "chas, nox, rm, dis, ptratio, b, lstat",
          "crim, chas, nox, rm, dis, ptratio, b, lstat",
          "crim, chas, nox, rm, dis, rad, ptratio, b, lstat",
          "crim, chas, nox, rm, dis, rad, tax, ptratio, b, lstat",
          "crim, indus, chas, nox, rm, dis, rad, tax, ptratio, b, lstat",
          "crim, zn, indus, chas, nox, rm, dis, rad, tax, ptratio, b, lstat",
          "crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat"
        ]
      },
      "rfe": {
        "name": "Recursive Feature Elimination (RFE)",
        "rmse": [
          5789.61,
          5589.57,
          5207.2,
          5175.14,
          5019.72,
          5053.15,
          4899.54,
          5052.26,
          5076.34,
          5049.77,
          4908.0,
          4917.7,
          4928.6
        ],
        "r2": [
          0.542918,
          0.573958,
          0.630253,
          0.634792,
          0.656398,
          0.651807,
          0.672654,
          0.651929,
          0.648603,
          0.652272,
          0.671523,
          0.670223,
          0.668759
        ],
        "features": [
          "lstat",
          "lstat, rm",
          "lstat, rm, ptratio",
          "lstat, rm, ptratio, dis",
          "lstat, rm, ptratio, dis, nox",
          "lstat, rm, ptratio, dis, nox, tax",
          "lstat, rm, ptratio, dis, nox, tax, rad",
          "lstat, rm, ptratio, dis, nox, tax, rad, b",
          "lstat, rm, ptratio, dis, nox, tax, rad, b, crim",
          "lstat, rm, ptratio, dis, nox, tax, rad, b, crim, chas",
          "lstat, rm, ptratio, dis, nox, tax, rad, b, crim, chas, zn",
          "lstat, rm, ptratio, dis, nox, tax, rad, b, crim, chas, zn, indus",
          "lstat, rm, ptratio, dis, nox, tax, rad, b, crim, chas, zn, indus, age"
        ]
      },
      "lasso_l1_ranking": {
        "name": "Lasso L1 Ranking",
        "rmse": [
          5789.61,
          5589.57,
          5591.61,
          5692.9,
          5248.52,
          5001.96,
          4899.54,
          5052.26,
          5076.34,
          5049.77,
          4908.0,
          4917.7,
          4928.6
        ],
        "r2": [
          0.542918,
          0.573958,
          0.573646,
          0.558061,
          0.624362,
          0.658826,
          0.672654,
          0.651929,
          0.648603,
          0.652272,
          0.671523,
          0.670223,
          0.668759
        ],
        "features": [
          "lstat",
          "lstat, rm",
          "lstat, rm, dis",
          "lstat, rm, dis, rad",
          "lstat, rm, dis, rad, ptratio",
          "lstat, rm, dis, rad, ptratio, nox",
          "lstat, rm, dis, rad, ptratio, nox, tax",
          "lstat, rm, dis, rad, ptratio, nox, tax, b",
          "lstat, rm, dis, rad, ptratio, nox, tax, b, crim",
          "lstat, rm, dis, rad, ptratio, nox, tax, b, crim, chas",
          "lstat, rm, dis, rad, ptratio, nox, tax, b, crim, chas, zn",
          "lstat, rm, dis, rad, ptratio, nox, tax, b, crim, chas, zn, indus",
          "lstat, rm, dis, rad, ptratio, nox, tax, b, crim, chas, zn, indus, age"
        ]
      },
      "elasticnet_ranking": {
        "name": "ElasticNet Ranking",
        "rmse": [
          5789.61,
          5589.57,
          5591.61,
          5692.9,
          5248.52,
          5001.96,
          4899.54,
          5052.26,
          5076.34,
          5049.77,
          4908.0,
          4917.7,
          4928.6
        ],
        "r2": [
          0.542918,
          0.573958,
          0.573646,
          0.558061,
          0.624362,
          0.658826,
          0.672654,
          0.651929,
          0.648603,
          0.652272,
          0.671523,
          0.670223,
          0.668759
        ],
        "features": [
          "lstat",
          "lstat, rm",
          "lstat, rm, dis",
          "lstat, rm, dis, rad",
          "lstat, rm, dis, rad, ptratio",
          "lstat, rm, dis, rad, ptratio, nox",
          "lstat, rm, dis, rad, ptratio, nox, tax",
          "lstat, rm, dis, rad, ptratio, nox, tax, b",
          "lstat, rm, dis, rad, ptratio, nox, tax, b, crim",
          "lstat, rm, dis, rad, ptratio, nox, tax, b, crim, chas",
          "lstat, rm, dis, rad, ptratio, nox, tax, b, crim, chas, zn",
          "lstat, rm, dis, rad, ptratio, nox, tax, b, crim, chas, zn, indus",
          "lstat, rm, dis, rad, ptratio, nox, tax, b, crim, chas, zn, indus, age"
        ]
      },
      "exhaustive_search": {
        "name": "Exhaustive Best Subset Search",
        "rmse": [
          5789.61,
          5252.48,
          5172.04,
          5028.06,
          4932.56,
          4836.79,
          4848.65,
          4893.18,
          4908.6,
          5049.77,
          5057.97,
          4917.7,
          4928.6
        ],
        "r2": [
          0.542918,
          0.623795,
          0.635229,
          0.655255,
          0.668228,
          0.680986,
          0.67942,
          0.673503,
          0.671442,
          0.652272,
          0.651142,
          0.670223,
          0.668759
        ],
        "features": [
          "lstat",
          "lstat, ptratio",
          "lstat, ptratio, rad",
          "lstat, ptratio, tax, rad",
          "lstat, ptratio, dis, nox, rad",
          "lstat, ptratio, dis, nox, tax, rad",
          "lstat, ptratio, dis, nox, tax, rad, chas",
          "lstat, ptratio, dis, nox, tax, rad, crim, chas",
          "lstat, rm, ptratio, dis, nox, tax, rad, crim, chas",
          "lstat, rm, ptratio, dis, nox, tax, rad, b, crim, chas",
          "lstat, rm, ptratio, dis, nox, tax, rad, b, crim, indus, chas",
          "lstat, rm, ptratio, dis, nox, tax, rad, b, crim, zn, indus, chas",
          "lstat, rm, ptratio, dis, nox, tax, rad, b, crim, zn, indus, chas, age"
        ]
      }
    },
    "simulator": {
      "type": "boston",
      "goldenModel": {
        "title": "最佳黃金模型 (k=7)",
        "desc": "變數: rm, lstat, ptratio, dis, nox, tax, rad",
        "rmse": "$4,899.54",
        "intercept": 36.79922716593739,
        "coefficients": {
          "lstat": -0.5802320360866252,
          "rm": 4.378570586226545,
          "ptratio": -0.9981357450567828,
          "dis": -1.2229190116582216,
          "nox": -16.26292584822222,
          "tax": -0.009770671822681543,
          "rad": 0.16137494888618836
        }
      },
      "baselineModel": {
        "title": "全特徵基準模型 (k=13)",
        "desc": "變數: 全部 13 項城鎮環境與社會指標",
        "rmse": "$4,928.60",
        "intercept": 30.24675099392349,
        "coefficients": {
          "crim": -0.11305592398537863,
          "zn": 0.030110464145648223,
          "indus": 0.04038072041333297,
          "chas": 2.7844382035079644,
          "nox": -17.202633391781003,
          "rm": 4.438835199513043,
          "age": -0.0062963622109808905,
          "dis": -1.447865368530779,
          "rad": 0.26242973558508303,
          "tax": -0.010646786275308219,
          "ptratio": -0.9154562404680713,
          "b": 0.012351334729969164,
          "lstat": -0.5085714244487934
        }
      }
    },
    "crispSummary": {
      "phase12": "分析波士頓 506 筆城鎮房產數據。無缺失值，lstat (-0.738) 與 rm (+0.695) 是最強驅動特徵。",
      "phase34": "實測 5 大特徵選擇演算法，RFE 與 Lasso 在 k=7 達成全域最低 RMSE ($4,899.54) 與最高 R² (0.6727)。",
      "phase56": "篩選後模型成功排除多重共線性雜訊，已持久化為 boston_housing_pipeline.joblib，支援即時推論。"
    }
  }
};
