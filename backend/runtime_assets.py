"""Portable, checksummed assets for the dashboard (no training datasets)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'
ARTIFACTS = {
    'quality_scores': '배터리 품질보증 모델/output/models/model_C_pca_t2_spe_이상점수.csv',
    'quality_folds': '배터리 품질보증 모델/output/models/model_C_pca_t2_spe_교차검증_fold별.csv',
    'demand_folds': '발주량 예측 모델/outputs/audited_v2/source_total/cv_metrics_by_fold.csv',
    'demand_errors': '발주량 예측 모델/outputs/audited_v2/source_total/final_holdout_part_metrics.csv',
    'maintenance_features': '배터리 예지보전 모델/outputs/track_b_final_v2/feature_importance.csv',
    'maintenance_operational': '배터리 예지보전 모델/outputs/track_b_final_v2/operational_metrics.csv',
    'maintenance_sensitivity': '배터리 예지보전 모델/outputs/track_b_final_v2/sensitivity_metrics.csv',
    'maintenance_forward': '배터리 예지보전 모델/outputs/track_b_final_v2/forward_time_metrics.csv',
    'maintenance_reverse': '배터리 예지보전 모델/outputs/track_b_final_v2/reverse_file_stress_metrics.csv',
    'quality_features': '배터리 품질보증 모델/output/models/model_A_random_forest_변수중요도.csv',
}


def quality_path(test):
    packed = RUNTIME / 'quality' / (test + '.csv.gz')
    return packed if packed.exists() else ROOT / '배터리 품질보증 모델/data/raw_data/test' / (test + '.csv')


def artifact_path(name):
    packed = RUNTIME / 'validation' / (name + '.csv')
    return packed if packed.exists() else ROOT / ARTIFACTS[name]


def verify_runtime():
    manifest = json.loads((RUNTIME / 'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = RUNTIME / item['file']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('실행 자료가 없거나 변경되었습니다: runtime/' + item['file'])
    return manifest
