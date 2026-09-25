# -*- coding: utf-8 -*-
"""
모델 5종 일괄 실행 및 비교

    A. Random Forest      지도학습 / 기준점        model_A_random_forest.py
    B. Isolation Forest   트리 기반 비지도 이상탐지  model_B_isolation_forest.py
    C. PCA (T²·SPE)       변수 간 상관 구조 이탈    model_C_pca_t2_spe.py
    E. T²+SPE 결합지표     두 통계량을 하나로 결합    model_E_combined_index.py
    D. LSTM Autoencoder   시계열 재구성 오차        model_D_lstm_autoencoder.py

데이터는 한 번만 읽어 모든 모델에 공유한다(모델마다 다시 읽으면 느리다).

실행
    python src/models/compare_all.py
    python src/models/compare_all.py --skip-lstm     # 딥러닝 제외하고 빠르게
"""

import os
import sys
import argparse
import importlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_data as C

MODULES = [
    ("model_A_random_forest", "A. Random Forest", "지도학습(라벨 사용)"),
    ("model_B_isolation_forest", "B. Isolation Forest", "비지도(트리 고립)"),
    ("model_C_pca_t2_spe", "C. PCA (T²·SPE)", "비지도(상관 구조)"),
    ("model_D_lstm_autoencoder", "D. LSTM Autoencoder", "비지도(시계열 재구성)"),
    ("model_E_combined_index", "E. T²+SPE 결합지표", "비지도(결합 통계량)"),
]


def main(skip_lstm=False):
    df_all, report, cell_z = C.load_dataset()
    C.save(report, "00_데이터_요약.csv")

    rows, per_file = [], []
    for mod_name, label, kind in MODULES:
        if skip_lstm and "lstm" in mod_name:
            print(f"\n(건너뜀: {label})")
            continue
        print("\n" + "=" * 78)
        print(f"  {label}  -  {kind}")
        print("=" * 78)
        mod = importlib.import_module(mod_name)
        res = mod.main(df_all=df_all, cell_z=cell_z)
        for r in res:
            r["분류"] = kind
        rows.extend(res)

        pf = os.path.join(C.OUT, f"model_{mod.PREFIX}_파일별.csv")
        if os.path.exists(pf):
            per_file.append(pd.read_csv(pf))

    res_df = pd.DataFrame(rows).round(4)
    C.save(res_df, "00_모델별_성능비교.csv")

    print("\n" + "=" * 78)
    print("  모델별 성능 비교")
    print("=" * 78)
    for split in ["학습", "테스트"]:
        sub = res_df[res_df["구분"] == split]
        if len(sub):
            print(f"\n[{split}]")
            print(sub[["모델", "분류", "Accuracy", "Precision", "Recall",
                       "F1-score", "TP", "FP", "FN"]].to_string(index=False))

    # 교차검증 결과 모으기 (단일 값)
    cv_rows = []
    for mod_name, label, kind in MODULES:
        pre = mod_name.replace("model_", "")
        f = os.path.join(C.OUT, f"model_{pre}_교차검증.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            d.insert(0, "모델", label.split(". ")[-1])
            cv_rows.append(d)
    if cv_rows:
        cv_all = pd.concat(cv_rows, ignore_index=True)
        print("\n[파일 단위 교차검증 (5-fold, Out-of-Fold 통합)]")
        print(cv_all.to_string(index=False))
        C.save(cv_all, "00_모델별_교차검증.csv")

    if per_file:
        pf = pd.concat(per_file, ignore_index=True)
        C.save(pf.sort_values(["파일", "모델"]), "00_모델별_파일별결과.csv")

        # 파일 x 모델 탐지율 표 : 어느 모델이 어떤 불량 유형을 잡는지
        ng = pf[pf["실제 이상"] > 0].copy()
        ng["탐지율"] = (ng["TP"] / ng["실제 이상"]).round(3)
        pivot = ng.pivot(index="파일", columns="모델", values="탐지율")
        print("\n[불량 파일별 탐지율 (TP / 실제 이상)]")
        print(pivot.to_string())
        C.save(pivot.reset_index(), "00_불량유형별_탐지율.csv")

        fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="YlGnBu",
                        zmin=0, zmax=1, aspect="auto",
                        title="불량 파일별 탐지율 - 모델마다 잡는 유형이 다르다")
        fig.write_html(os.path.join(C.OUT, "00_탐지율_비교.html"), include_plotlyjs="cdn")
        try:
            fig.write_image(os.path.join(C.OUT, "00_탐지율_비교.png"),
                            width=1100, height=500, scale=2)
        except Exception:
            pass

    # 성능 비교 막대
    m = res_df[res_df["구분"] == "테스트"].melt(
        id_vars="모델", value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
        var_name="지표", value_name="값")
    fig = px.bar(m, x="지표", y="값", color="모델", barmode="group", text_auto=".3f",
                 title="모델별 성능 비교 (테스트 5개 시험, 정상 기준 10개 파일 학습)")
    fig.add_hline(y=0.7, line_dash="dash", annotation_text="기획서 목표: Precision 0.7")
    fig.update_yaxes(range=[0, 1.08])
    fig.write_html(os.path.join(C.OUT, "00_모델비교.html"), include_plotlyjs="cdn")
    try:
        fig.write_image(os.path.join(C.OUT, "00_모델비교.png"),
                        width=1200, height=650, scale=2)
    except Exception as e:
        print("(PNG 저장 생략:", e, ")")

    print("\n산출물 위치:", C.OUT)
    return res_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lstm", action="store_true")
    a = ap.parse_args()
    main(skip_lstm=a.skip_lstm)
