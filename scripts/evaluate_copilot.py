"""Reproducible real retrieval/LLM scenarios; --live explicitly spends Gemini quota."""
import argparse
import json
import re
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.copilot import Copilot
from backend.data import Repository


def validate(name,result):
    text=result.get('text','').replace(',','')
    facts=result.get('context',{})
    if result.get('status')!='answered' or not result.get('citations'):return False
    if name in ('part_lookup','followup'):
        return facts.get('part')=='Part 92' and all(value in text for value in ('Part 92','642.29','102','540.29'))
    if name=='ranked_parts':
        return re.findall(r'Part\s+\d+',text)==[r['part'] for r in facts['matches'][:3]] and '-284.86' in text
    if name=='zero_power':
        return facts.get('zeroPowerRows')==39 and all(value in text for value in ('39','EVT-003','78','116'))
    if name=='defect_type':
        return '용량' in text and '확정' in text and bool(re.search('아닙|없|불가',text))
    if name=='model_rationale':
        return all(value in text for value in ('7일','34.66','69.70','35.70','ERP')) and any(c['id'].startswith('S') and c['id']!='SCREEN' for c in result['citations'])
    return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--case',help='Run one named scenario (followup requires part_lookup).')
    args=parser.parse_args()
    copilot=Copilot(Repository())
    scenarios=[
        ('part_lookup','품질 화면이지만 Part 92의 예측량과 기존 계획을 찾아줘',{'track':'quality'},['Part 92']),
        ('ranked_parts','계획과 차이가 큰 부품 3개를 찾아줘',{'track':'overview'},['Part 92']),
        ('zero_power','WeldingTest_04_NG에서 출력 0W 구간과 점검 이벤트를 찾아줘',{'track':'quality'},['39']),
        ('defect_type','Test08의 불량 유형 근거와 확정 진단 가능 여부를 알려줘',{'track':'demand'},['용량']),
        ('model_rationale','왜 발주량 기본 모델이 7일 이동평균이야? 재학습 검증 결과와 한계를 설명해줘',{'track':'demand'},['7']),
        ('followup','그 부품의 기존 계획과 차이를 다시 알려줘',{'track':'quality'},['Part 92'])]
    results=[];first=None
    if args.case:
        scenarios=[s for s in scenarios if s[0]==args.case or (args.case=='followup' and s[0]=='part_lookup')]
        if not scenarios:parser.error('unknown scenario')
    folder=ROOT/'.local/copilot-evaluations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder.mkdir(parents=True)
    for index,(name,question,context,expected) in enumerate(scenarios):
        if args.live and index:time.sleep(22)
        start=time.monotonic()
        try:
            if args.live:
                result=copilot.answer(question,context,[{'role':'assistant',**first}] if name=='followup' and first else [])
                if name=='part_lookup':first=result
                passed=validate(name,result)
            else:
                domain='demand' if name in ['part_lookup','ranked_parts','model_rationale','followup'] else 'maintenance' if name=='zero_power' else 'quality'
                result={'retrieved':copilot.retrieve(question+' '+domain)}
                passed=any(c['source'].startswith(domain) for c in result['retrieved'])
            row={'id':name,'question':question,'passed':bool(passed),'seconds':round(time.monotonic()-start,2),'result':result}
        except Exception as exc:
            row={'id':name,'question':question,'passed':False,'error':getattr(exc,'detail',type(exc).__name__)}
        results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='result'},ensure_ascii=False),flush=True)
        (folder/'results.json').write_text(json.dumps({'live':args.live,'results':results},ensure_ascii=False,indent=2),encoding='utf-8')
        if 'error' in row:break
    print(json.dumps({'report':str(folder/'results.json'),'passed':sum(r['passed'] for r in results),'executed':len(results),'planned':len(scenarios)}),flush=True)
    if len(results)!=len(scenarios) or not all(r['passed'] for r in results):sys.exit(1)


if __name__=='__main__':main()
