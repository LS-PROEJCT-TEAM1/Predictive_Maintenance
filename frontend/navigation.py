"""Allowlisted URL context. Missing parameters preserve the user's selections."""
import re
from urllib.parse import parse_qs


def navigation_context(track, search, meta):
    query = {k:v[-1] for k,v in parse_qs((search or '').lstrip('?')).items() if v}
    result = {}
    options = {
        'demand':{'date':('d-date',meta['dates']),'part':('d-part',['ALL',*meta['parts']]),'model':('d-model',meta['demandModels'])},
        'maintenance':{'run':('m-run',meta['runs']),'supervised':('m-sup',meta['supervised']),'unsupervised':('m-unsup',meta['unsupervised'])},
        'quality':{'test':('q-test',meta['tests'])}}
    for param,(component,valid) in options.get(track,{}).items():
        if query.get(param) in valid:
            result[component] = query[param]
    cell = query.get('cell','')
    if track=='quality' and re.fullmatch(r'M(0[1-9]|1[0-6])CV(0[1-9]|1[01])',cell):
        result['q-cell'] = cell
    event = query.get('event','')
    result['event'] = event if track=='maintenance' and len(event)<=250 and event.startswith(query.get('run','')+':') else None
    result['tab'] = query.get('tab')
    result['resetLocalFilters'] = query.get('tab')=='review' and bool(result.keys()-{'tab','event'})
    return result
