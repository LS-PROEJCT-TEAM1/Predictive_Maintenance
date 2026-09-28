"""Predeclared multi-unit and sensitivity evidence; never changes model selection."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.maintenance_policy import POLICY, alarm_regions


def counts(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp, fn, tn = [int(a.sum()) for a in [y&p, ~y&p, y&~p, ~y&~p]]
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                fpr=fp/(fp+tn) if fp+tn else None)


def operational_metrics(frame, prediction):
    work = frame.copy().reset_index(drop=True)
    work['prediction'] = np.asarray(prediction, int)
    result = {}
    for unit, groups in [('cycle', ['source_file', 'group_id']), ('file', ['source_file'])]:
        agg = work.groupby(groups)[['label', 'prediction']].max()
        result.update({f'{unit}_{k}': v for k, v in counts(agg.label, agg.prediction).items()})
        result[f'{unit}_count'] = len(agg)
    alarm_count = false_alarms = 0
    hours = normal_hours = 0.
    for source, block in work.groupby('source_file', sort=False):
        block = block.sort_values('source_row')
        rows = [dict(row=int(r.source_row), time=str(r.WorkingTime), prediction=int(r.prediction), source=source)
                for r in block.itertuples()]
        regions = alarm_regions(rows, POLICY['gapSeconds'])
        labels = block.label.to_numpy()
        alarm_count += len(regions)
        false_alarms += sum(not labels[a:b+1].any() for a,b in regions)
        gaps = pd.to_datetime(block.WorkingTime).diff().dt.total_seconds()
        valid = gaps.gt(0) & gaps.le(POLICY['gapSeconds']) & block.source_row.diff().eq(1)
        normal = block.label.eq(0) & block.label.shift(1).eq(0)
        hours += float(gaps[valid].sum())/3600
        normal_hours += float(gaps[valid & normal].sum())/3600
    result.update(alarm_count=alarm_count, false_alarm_count=false_alarms,
                  observed_hours=hours, normal_observed_hours=normal_hours,
                  false_alarms_per_normal_observed_hour=false_alarms/normal_hours if normal_hours else None,
                  policy_version=POLICY['version'])
    return result


def all_operational(models, validation, test):
    output = []
    for split, frame in [('validation', validation), ('locked_test', test)]:
        predictions = {d.name:d.predict(frame) for d in models}
        for detector in models:
            output.append(dict(model=detector.name, split=split, family=detector.family,
                               **operational_metrics(frame, predictions[detector.name])))
        for a in [d for d in models if d.family == 'supervised']:
            for b in [d for d in models if d.family == 'unsupervised']:
                output.append(dict(model=f'{a.name} OR {b.name}', supervised=a.name, unsupervised=b.name,
                                   split=split, family='combination',
                                   **operational_metrics(frame, predictions[a.name] | predictions[b.name])))
    return pd.DataFrame(output)


def supplementary(m, models, historical, hist_train, hist_cal, tests, main_test):
    """All settings fixed in TRAINING_PLAN_V3.md before retraining."""
    sensitivity = []
    # Supervised group-boundary sensitivity; keep the main normal reference fixed.
    for fraction in [.5, .6]:
        train, val, _ = m.split_development_by_cycle([tests['WeldingTest_01_OK'],tests['WeldingTest_03_NG']], fraction)
        candidates, _ = m.fit_models(hist_train, hist_cal, train, val, include_unsupervised=False)
        for d in candidates:
            metrics, _ = m.calculate_metrics(main_test, d.predict(main_test))
            sensitivity.append(dict(experiment='development_fraction', setting=fraction, model=d.name,
                                    used_for_selection=False, validation_positive_rows=int(val.label.sum()), **metrics))
    for d in models:
        if d.family != 'unsupervised':
            continue
        cal_scores, scores = d.score(hist_cal), d.score(main_test)
        for q in [.995, .999, .9995]:
            threshold = float(np.quantile(cal_scores, q, method='higher'))
            pred = (scores >= threshold).astype(int)
            metrics, _ = m.calculate_metrics(main_test, pred)
            sensitivity.append(dict(experiment='normal_quantile', setting=q, threshold=threshold,
                                    model=d.name, used_for_selection=False, **metrics))
    cutoff = min(f.WorkingTime.min() for f in tests.values())
    eligible = historical.groupby('group_id').WorkingTime.max()
    prior = historical[historical.group_id.isin(eligible[eligible < cutoff].index)]
    train, cal, split = m.split_historical_normal(prior)
    ref = m.phase_reference(train)
    threshold = float(np.quantile(m.make_features(cal, ref).PhaseAbsZ, .999, method='higher'))
    forward = []
    for source, frame in tests.items():
        pred = (m.make_features(frame, ref).PhaseAbsZ.to_numpy() >= threshold).astype(int)
        metric, _ = m.calculate_metrics(frame, pred)
        forward.append(dict(source_file=source, model='RobustPhaseZ', protocol='normal_only_forward_time',
                            fit_end=str(train.WorkingTime.max()), calibration_end=str(cal.WorkingTime.max()),
                            test_start=str(frame.WorkingTime.min()), threshold=threshold,
                            used_for_selection=False, **metric, **operational_metrics(frame, pred)))
    return pd.DataFrame(sensitivity), pd.DataFrame(forward), split


def write_report(m, manifest, metrics, operating, sensitivity, forward):
    output = m.OUTPUT_DIR
    winner = manifest['winner_selected_on_validation']
    locked = metrics[metrics['split'].eq('locked_test')]
    columns = ['model','precision','recall','f1','fp','fn','event_total','event_recall']
    text = f'''# 예지보전 재검증 보고서 v3

선택 기준모델: **{winner}**. 운영 지도/비지도 조합은 validation에서 각각 선택 후 OR로 결합한다.
평가 성격: **재사용 파일 회고 평가**. 이전 실험에서 본 자료이며 새로운 외부 검증이 아니다.
현재 이상 탐지이며 미래 고장 시간·RUL을 예측하지 않는다.

## 데이터 처리

사이클 내부 시간 오류 {manifest['historical_split']['excluded_time_cycles']}개와 0 출력 사이클 {manifest['historical_split']['excluded_zero_power_cycles']}개를 격리했다.
원본 ID를 유지하며 사이클 시작 시각으로 정렬했다. 정상 fit/calibration 경계가 분리됨을 검사했다.
주 평가의 정상 학습 자료는 02_OK보다 미래 자료를 포함한다. 전체를 미래 시점 검증으로 부르지 않는다.
별도 normal_only_forward_time 결과만 모든 시험 이전 자료로 적합·보정했다.

## 행 단위 회고 시험

{m.markdown_table(locked, columns)}

## 사이클 및 운영 경보

{m.markdown_table(operating[operating['split'].eq('locked_test')], ['model','cycle_precision','cycle_recall','cycle_f1','cycle_fpr','false_alarm_count'])}

사이클은 한 행 이상 이상이면 양성이다. 경보는 첫 양성 행에서 시작하고 정상 행 또는 120초 초과 수집 공백에서 종료한다.
관측시간당 오경보는 0~120초의 유효한 인접 간격만 사용한 보조 지표이며 실제 가동시간이 아니다.
사이클 FPR의 현장 승인 한도는 미정이다. 행 FPR 1%와 같은 지표로 비교하지 않는다.

## 추가 시간 검증: 정상 전용 기준모델

{m.markdown_table(forward, ['source_file','precision','recall','f1','fp','fn'])}

별도 기준모델의 시간 검증이며 주 모델 선택에는 사용하지 않았다. 고장 종류와 설비 일반화의 외부 검증을 대체하지 않는다.

## 한계와 선택

지도 validation 이상은 3건, 회고 시험 이상은 1개의 연속 사건이다. 행/사이클 수를 독립 사건 수로 해석하지 않는다.
동률은 사전에 고정한 단순성 순서를 사용했다. 측정 학습시간이나 역방향 시험 성적을 선택에 사용하지 않았다.
개발 분할·정상 분위수 민감도는 sensitivity_metrics.csv, 반대 파일 방향은 reverse_file_stress_metrics.csv에 보존했다.
반대 방향 개발은 NG04의 단일 사건을 나누므로 사건 독립 검증이 아니다.
외부 신규 사건·현장 SOP·단위 및 점검 비용 기준은 추가 확보가 필요하다.

## 재현성

run_manifest.json에 계획·코드·입력·모델 해시, 환경 버전, 선택 시각을 기록했다.
selection.json은 회고 시험 모델 추론 전에 저장했다. 학습 산출물 경로는 기존 연동을 위해 track_b_final_v2를 유지하며 모델 버전은 {m.PIPELINE_VERSION}다.
'''
    (output/'final_validation_report.md').write_text(text, encoding='utf-8')
    plan = m.PROJECT_ROOT/'docs/TRAINING_PLAN_V3.md'
    (output/'training_plan.md').write_text(plan.read_text(encoding='utf-8'), encoding='utf-8')
    (output/'data_quality_report.md').write_text('# 데이터 처리 v3\n\n' + json.dumps(manifest['historical_split'],ensure_ascii=False,indent=2) + '\n\n원본 데이터는 수정하지 않음. 시간 역전은 원본 data_quality.csv에 그대로 보고함.\n',encoding='utf-8')
