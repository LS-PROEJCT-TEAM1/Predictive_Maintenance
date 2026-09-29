"""Calibrate saved defect metrics by process; no AI inference or Firestore writes."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('capacity', 'weld', 'wire', 'sensor')
MODES = ('chg', 'dchg')
SPLIT_SEED = 'pack-defects-v2-20260929'


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def build(packs):
    reference = {}
    for ident, doc in sorted(packs.items()):
        parts = ident.rsplit('_', 1)
        if len(parts) != 2 or not parts[0].isdigit() or parts[1] not in MODES:
            continue
        metrics = doc.get('metrics') or {}
        if doc.get('source') == 'train' and doc.get('grade') == 'OK' and all(finite(metrics.get(k)) for k in KEYS):
            reference[ident] = {'pack': parts[0], 'process': parts[1], 'metrics': {k:metrics[k] for k in KEYS},
                                'source': doc['source'], 'grade': doc['grade'], 'n_rows': doc.get('n_rows')}
    groups = sorted({r['pack'] for r in reference.values()})
    if len(groups) < 20 or any(f'{g}_{m}' not in reference for g in groups for m in MODES):
        raise ValueError('Need at least 20 complete normal-reference charge/discharge pack pairs.')
    ranked = sorted(groups, key=lambda g: hashlib.sha256(f'{SPLIT_SEED}|{g}'.encode()).hexdigest())
    n_eval = max(1, round(len(groups)*0.3))
    evaluation, fitting = sorted(ranked[:n_eval], key=int), sorted(ranked[n_eval:], key=int)
    thresholds, stats, evaluation_results = {}, {}, {}
    for mode in MODES:
        thresholds[mode], stats[mode], evaluation_results[mode] = {}, {}, {}
        for key in KEYS:
            values = [reference[f'{g}_{mode}']['metrics'][key] for g in fitting]
            mean, std = statistics.mean(values), statistics.pstdev(values)
            threshold = mean+3*std
            thresholds[mode][key] = threshold
            stats[mode][key] = {'n':len(values), 'mean':mean, 'stdPopulation':std}
            flagged = [f'{g}_{mode}' for g in evaluation if reference[f'{g}_{mode}']['metrics'][key] > threshold]
            evaluation_results[mode][key] = {'n':len(evaluation), 'flagged':len(flagged),
                                            'rate':len(flagged)/len(evaluation), 'ids':flagged}
        any_flags = sorted(set(i for result in evaluation_results[mode].values() for i in result['ids']))
        evaluation_results[mode]['any'] = {'n':len(evaluation), 'flagged':len(any_flags),
                                            'rate':len(any_flags)/len(evaluation), 'ids':any_flags}
    payload = {'schemaVersion':1, 'method':'mean_plus_3_population_std', 'comparison':'strict_greater_than',
               'metricScope':{'capacity':'maximum_adjacent_sample_voltage_change',
                              'weld':'mean_minus_min_voltage_at_saved_AI_peak',
                              'wire':'max_adjacent_cell_gap_at_saved_AI_peak',
                              'sensor':'max_temperature_median_deviation_at_saved_AI_peak'},
               'units':{'capacity':'mV','weld':'mV','wire':'mV','sensor':'C'},
               'thresholdsByProcess':thresholds, 'fittingStatistics':stats,
               'split':{'seed':SPLIT_SEED,'unit':'pack_number','fittingPacks':fitting,'evaluationPacks':evaluation},
               'evaluation':evaluation_results, 'reference':reference,
               'limitations':['Reference grade OK is not a confirmed four-type defect label.',
                              'Internal held-out reference exceedance rates, not external defect accuracy.',
                              'Full time-series recomputation unavailable for 92 of the 102 source records.',
                              'No threshold tuning or refitting after viewing evaluation outcomes.']}
    version = hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    return {'version':'pack-defects-by-process-'+version, **payload}


def report(data):
    split = data['split']
    lines = ['# 충전·방전별 불량 의심 지표 임계값 재산출', '',
             f"정책: `{data['version']}`", '',
             f"정상 기준 51개 팩 중 산출 {len(split['fittingPacks'])}개, 평가 {len(split['evaluationPacks'])}개. 같은 팩의 충전·방전은 같은 분할에 배치했습니다.",
             '기준은 평균 + 3 × 모집단 표준편차이며, 비교에는 반올림 전 값을 사용합니다. 평가 결과를 보고 기준을 조정하지 않았습니다.', '',
             '| 유형 | 충전 | 방전 | 단위 |', '|---|---:|---:|---|']
    names = {'capacity':'용량불량','weld':'용접불량','wire':'센서와이어불량','sensor':'센서불량'}
    for k in KEYS:
        lines.append(f"| {names[k]} | {data['thresholdsByProcess']['chg'][k]:.2f} | {data['thresholdsByProcess']['dchg'][k]:.2f} | {data['units'][k]} |")
    lines += ['', '## 분리된 정상 기준 평가', '', '| 공정 | 용량 | 용접 | 와이어 | 센서 | 하나 이상 |', '|---|---:|---:|---:|---:|---:|']
    for mode in MODES:
        results = data['evaluation'][mode]
        values = [f"{results[k]['flagged']}/{results[k]['n']} ({results[k]['rate']:.1%})" for k in (*KEYS,'any')]
        lines.append('| '+('충전' if mode=='chg' else '방전')+' | '+' | '.join(values)+' |')
    lines += ['', '정상 기준으로 분류된 자료의 초과 비율입니다. 현장 확정 불량 정답이 없으므로 실제 원인 분류 정확도나 NG 검출률을 의미하지 않습니다.',
              '용량 지표만 전체 시계열의 최대 변화량이며 나머지 세 지표는 저장된 AI 최대 점수 시점 값입니다. 현재 원본은 10건만 확보되어 전체 시계열 지표로 재산출하지 않았습니다.',
              '기존 AI OK/NG와 Firestore 원본 문서는 변경하지 않습니다. 화면의 의심 유형은 이 정책을 적용한 별도 규칙 결과입니다.', '',
              '산출 팩: '+', '.join(split['fittingPacks']), '평가 팩: '+', '.join(split['evaluationPacks']), '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='JSON with packs, or a previous calibration artifact with reference')
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding='utf-8'))
    data = build(source.get('packs', source.get('reference', {})))
    output = ROOT/'runtime/quality/defect_calibration.json'
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (ROOT/'QUALITY_THRESHOLD_REPORT.md').write_text(report(data),encoding='utf-8')
    print(json.dumps({'version':data['version'],'thresholds':data['thresholdsByProcess'],'evaluation':data['evaluation']},ensure_ascii=True))


if __name__ == '__main__':
    main()
