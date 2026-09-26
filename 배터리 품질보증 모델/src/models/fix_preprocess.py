# -*- coding: utf-8 -*-
"""
전처리 복구 — removeConstant가 날려버린 팩 되살리기

문제
    기존 전처리(가이드북 [코드 19] removeConstant)는 '값 종류가 1개뿐인 컬럼'을
    상수로 보고 제거한다. 그런데 거의 정지 상태로 측정된 팩은 셀 전압이
    4mV 안에서만 움직여, 208개 컬럼이 전부 상수로 판정되어 삭제된다.
    결과 파일에는 줄바꿈만 남는다.

    예) 1017_chg : 원본 7,246행 × 231열, 셀 전압 범위 [3.611, 3.615]
                   → 전처리 결과 0열

해결
    removeConstant를 거치지 않고 컬럼명 패턴으로 208개를 직접 추출한다.
    변화 폭이 작은 것은 '정상적으로 잔잔한 팩'이지 버릴 데이터가 아니다.

실행
    python src/models/fix_preprocess.py              # 깨진 파일만 복구
    python src/models/fix_preprocess.py --all        # 전체 재생성
    python src/models/fix_preprocess.py --check      # 점검만 (쓰기 없음)
"""

import os
import re
import sys
import glob
import argparse

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_data as C

RAW_TRAIN = os.path.join(C.ROOT, "data", "raw_data", "train")


def is_broken(path):
    """전처리 결과가 비어 있는지 확인."""
    if not os.path.exists(path):
        return True
    try:
        d = pd.read_csv(path, nrows=5)
        return d.shape[1] == 0
    except Exception:
        return True


def extract_208(src):
    """원본에서 셀 전압 176 + 모듈 온도 32만 컬럼명으로 직접 추출."""
    d = pd.read_csv(src)
    cv = [c for c in d.columns if re.match(r"M\d+CV\d+$", c)]
    tp = [c for c in d.columns if re.match(r"M\d+T\d+$", c)]
    if not cv or not tp:
        return None, len(cv), len(tp)
    return d[cv + tp].apply(pd.to_numeric, errors="coerce"), len(cv), len(tp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="깨지지 않은 파일도 재생성")
    ap.add_argument("--check", action="store_true", help="점검만 하고 쓰지 않음")
    a = ap.parse_args()

    raws = sorted(glob.glob(os.path.join(RAW_TRAIN, "*.csv")))
    if not raws:
        sys.exit(f"[오류] 원본이 없습니다: {RAW_TRAIN}")

    targets = []
    for src in raws:
        name = os.path.basename(src)
        dst = os.path.join(C.PRE_TRAIN, name)
        if a.all or is_broken(dst):
            targets.append((src, dst, name))

    print(f"원본 {len(raws)}개 중 대상 {len(targets)}개\n")
    if not targets:
        print("복구할 파일이 없습니다.")
        return

    fixed, failed = [], []
    for src, dst, name in targets:
        data, n_cv, n_tp = extract_208(src)
        if data is None:
            failed.append((name, f"컬럼 불일치 (셀 {n_cv} / 온도 {n_tp})"))
            print(f"  [실패] {name:16s} 셀 {n_cv} / 온도 {n_tp}")
            continue
        rng = float(data.iloc[:, :n_cv].max().max() - data.iloc[:, :n_cv].min().min())
        print(f"  [복구] {name:16s} {data.shape[0]:6,d}행 × {data.shape[1]}열  "
              f"셀전압 변화폭 {rng*1000:6.1f} mV")
        if not a.check:
            data.to_csv(dst, index=False)
        fixed.append(name)

    print(f"\n{'점검' if a.check else '복구'} 완료: {len(fixed)}개"
          + (f" / 실패 {len(failed)}개" if failed else ""))
    if not a.check and fixed:
        print("\n다음 단계: python src/models/extract_firebase.py")


if __name__ == "__main__":
    main()
