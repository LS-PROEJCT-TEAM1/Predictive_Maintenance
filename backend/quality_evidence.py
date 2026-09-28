"""Inspection evidence, not a trained four-class fault classifier."""
import numpy as np


DEFECTS = [
    {"id": "capacity", "label": "용량 불량", "symptom": "충·방전 중 특정 셀 전압이 빠르게 상승하거나 하락하는 현상",
     "check": "선택 셀과 모듈 평균의 추세, 시험 말기의 전압 벌어짐을 확인하세요.",
     "source": "가이드북 PDF 5쪽 · Test08 예제 54쪽"},
    {"id": "welding", "label": "용접불량", "symptom": "셀 전압 미측정 또는 전압 저하가 나타나는 현상",
     "check": "전압 누락·저하 위치와 접합부 점검 결과를 함께 확인하세요. 신호만으로 원인을 확정할 수 없습니다.",
     "source": "가이드북 PDF 6쪽 · 시험별 확정 유형 미확인"},
    {"id": "sensor", "label": "센서불량", "symptom": "측정값이 비정상적으로 높거나 낮게 출력되는 현상",
     "check": "원본값의 물리 범위 이탈과 결측을 확인하고 센서·BMS 설정을 점검하세요. 보정값은 원인 확인 근거를 가릴 수 있습니다.",
     "source": "가이드북 PDF 7쪽 · Test06의 유형은 확인 필요"},
    {"id": "wire", "label": "센서와이어불량", "symptom": "인접 셀 사이에 비정상적인 전압 차이가 생기는 현상",
     "check": "인접 셀 전압과 측정 연결 상태를 확인하세요. 자료에서는 ‘센싱 와이어’로도 표기합니다.",
     "source": "가이드북 PDF 6·57쪽 · Test07 본문/코드 주석 불일치"},
]


def number(value):
    return float(value) if np.isfinite(value) else None


def snapshot(raw, clean, index, basis):
    """Keep raw values and quality flags even when the display is cleaned."""
    cells, temperatures = [], []
    for column in raw.columns:
        voltage = "CV" in column
        lo, hi = (2, 5) if voltage else (-20, 100)
        original, corrected = number(raw.iloc[index][column]), number(clean.iloc[index][column])
        invalid = original is None or not lo <= original <= hi
        item = {"id": column, "module": column[:3], "channel": column[3:], "raw": original,
                "clean": corrected, "value": original if basis == "raw" else corrected,
                "invalid": invalid, "reason": "결측·비유한값" if original is None else "물리 범위 이탈" if invalid else "범위 내"}
        (cells if voltage else temperatures).append(item)

    def stats(items):
        values = [p["value"] for p in items if p["value"] is not None]
        return {"min": min(values) if values else None, "max": max(values) if values else None,
                "mean": float(np.mean(values)) if values else None,
                "range": max(values)-min(values) if values else None,
                "invalidCount": sum(p["invalid"] for p in items), "available": len(values)}

    return {"index": index, "cells": cells, "temperatures": temperatures,
            "voltage": stats(cells), "temperature": stats(temperatures)}


def defect_evidence(test, snap):
    items = []
    for definition in DEFECTS:
        item = dict(definition)
        item.update(status="유형 미확정", sourceMatch=False, evidence="현재 모델은 정상/이상 탐지이며 불량 원인을 분류하지 않습니다.")
        if test == "Test08_NG_chg" and item["id"] == "capacity":
            item.update(status="자료상 사례", sourceMatch=True, evidence="가이드북의 Test08 예제에 용량 불량으로 표기되어 있습니다. AI 원인 확정은 아닙니다.")
        elif test == "Test07_NG_dchg" and item["id"] == "wire":
            item.update(status="자료 확인 필요", sourceMatch=True, evidence="Test07 본문은 센싱 와이어 불량, 코드 주석은 용량 불량으로 표기되어 원자료 확인이 필요합니다.")
        elif item["id"] == "sensor":
            count = snap["voltage"]["invalidCount"] + snap["temperature"]["invalidCount"]
            item["evidence"] = f"선택 시점 원본의 결측·물리 범위 이탈 {count}개 채널. 센서 고장의 확정 진단은 아닙니다."
            if count:
                item["status"] = "측정값 확인 필요"
        items.append(item)
    return items
