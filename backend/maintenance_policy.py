"""Versioned causal alarm aggregation shared by model evaluation and dashboard."""
from datetime import datetime

POLICY = {
    'version': 'maintenance-alarm-v3', 'combination': 'selected_supervised_OR_selected_unsupervised',
    'cycle': 'any_alarm_in_39_rows', 'file': 'any_alarm_cycle',
    'start': 'first_positive_row', 'clear': 'first_normal_row',
    'gapSeconds': 120, 'minimumRows': 1, 'cooldownRows': 0,
    'approval': 'research_only; cycle/field false-alarm cost limit not approved',
}


def alarm_regions(rows, gap_seconds=120):
    """Return inclusive list offsets; never bridge missing rows or acquisition gaps."""
    found, start = [], None
    for i, row in enumerate(rows):
        boundary = i > 0 and (
            row['row'] != rows[i-1]['row'] + 1
            or row.get('source', '') != rows[i-1].get('source', '')
            or not 0 < (datetime.fromisoformat(str(row['time'])) -
                        datetime.fromisoformat(str(rows[i-1]['time']))).total_seconds() <= gap_seconds)
        if start is not None and (boundary or not row['prediction']):
            found.append((start, i-1))
            start = None
        if row['prediction'] and start is None:
            start = i
    if start is not None:
        found.append((start, len(rows)-1))
    return found
