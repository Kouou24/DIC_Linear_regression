"""
多元線性迴歸與特徵選擇主執行入口 (Main Workflow Entrypoint)
支援 50 Startups 與 Boston Housing 雙資料集 CRISP-DM 流程與特徵演進分析
"""

import sys
import argparse
import subprocess

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def run_command(desc, cmd):
    print("\n" + "=" * 70)
    print(f"  執行: {desc}")
    print("=" * 70)
    result = subprocess.run([sys.executable] + cmd, check=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="多元線性迴歸與特徵選擇專案主程式")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "startups", "boston"],
        default="all",
        help="選擇執行的資料集 (all: 全部, startups: 50 Startups, boston: Boston Housing)"
    )
    parser.add_argument(
        "--progression",
        action="store_true",
        help="是否同時重新計算特徵選擇進程 (feature_progression_analysis.py)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  CRISP-DM 多元線性迴歸與特徵選擇管線啟動")
    print(f"  選擇資料集: {args.dataset}")
    print("=" * 70)

    if args.dataset in ["all", "startups"]:
        run_command("50 Startups CRISP-DM 流程", ["50_startups_crisp_dm.py"])

    if args.dataset in ["all", "boston"]:
        run_command("Boston Housing CRISP-DM 流程", ["boston_housing_crisp_dm.py"])

    if args.progression or args.dataset == "all":
        run_command("5 大演算法特徵演進分析與數據產出", ["feature_progression_analysis.py"])

    print("\n" + "=" * 70)
    print("  所有選定流程皆已順利執行完畢！")
    print("  可開啟 index.html 透過頂部下拉選單切換檢視 50 Startups 與 Boston Housing 的互動分析！")
    print("=" * 70)


if __name__ == "__main__":
    main()

