"""Versioned process-specific rules for suspected defects, separate from AI OK/NG."""
import json
import math
from pathlib import Path

POLICY_FILE = Path(__file__).resolve().parents[1]/'runtime/quality/defect_calibration.json'
KEYS = ('capacity','weld','wire','sensor')


def load_policy():
    policy = json.loads(POLICY_FILE.read_text(encoding='utf-8'))
    if policy.get('schemaVersion') != 1 or policy.get('method') != 'mean_plus_3_population_std':
        raise ValueError('불량 의심 유형의 임계값 정책을 확인하세요.')
    for mode in ('chg','dchg'):
        thresholds = policy['thresholdsByProcess'][mode]
        for key in KEYS:
            value = thresholds[key]
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value < 0:
                raise ValueError('불량 의심 유형 임계값이 올바르지 않습니다.')
    return policy


def apply_policy(row, policy):
    thresholds = policy['thresholdsByProcess'][row['process']]
    metrics = row.get('metrics') or {}
    flags = {}
    for key in KEYS:
        value = metrics.get(key)
        valid = not isinstance(value,bool) and isinstance(value,(float,int)) and math.isfinite(value)
        flags[key] = value > thresholds[key] if valid else None
    return {**row, 'storedDefectFlags':row.get('flags'), 'flags':flags,
            'defectPolicyVersion':policy['version']}
