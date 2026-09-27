"""이상 탐지 탭에서 Callback 이 채워 넣는 화면 조각 (지표 카드·모듈 카드·혼동행렬 등)."""

from dash import html

from common.layout import make_empty_message, make_metric_card
from common.styles import (
    CARD_TITLE_STYLE, FONT_NUM, GRAY, INK, LIGHT, TEAL_050, TEAL_200,
    TEAL_500, TEAL_600, WHITE,
)
from tabs.anomaly import data

NOT_APPLICABLE = "해당 없음"

BADGE_CLASS = {data.SUSPECT: "badge-suspect", data.RECHECK: "badge-recheck", data.GOOD: "badge-good"}
BADGE_STYLE = {
    data.SUSPECT: {"backgroundColor": TEAL_500, "color": INK},
    data.RECHECK: {"backgroundColor": TEAL_200, "color": INK},
    data.GOOD: {"backgroundColor": LIGHT, "color": GRAY},
}


def fmt_int(value):
    return f"{int(value):,}"


# ---------------------------------------------------------------- 지표 카드 6개

def make_metric_cards(metrics):
    if metrics is None:
        return [html.Div(make_empty_message(), style={"gridColumn": "1 / -1"})]

    total = metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
    actual_anomaly = metrics["tp"] + metrics["fn"]
    actual_normal = metrics["tn"] + metrics["fp"]
    predicted = metrics["tp"] + metrics["fp"]
    missed = int(metrics["event_total"] - metrics["event_detected"])

    # OK 파일처럼 실제 이상 행이 없으면 Precision·Recall·F1 은 정의되지 않는다
    has_anomaly = actual_anomaly > 0
    score_color = INK if has_anomaly else GRAY

    def score(value):
        return f"{value:.4f}" if has_anomaly else NOT_APPLICABLE

    def score_sub(text):
        return text if has_anomaly else "실제 이상 없음"

    return [
        make_metric_card("Accuracy", f"{metrics['accuracy']:.4f}",
                         f"정답 {fmt_int(metrics['tn'] + metrics['tp'])} / {fmt_int(total)}"),
        make_metric_card("Precision", score(metrics["precision"]),
                         score_sub(f"TP {fmt_int(metrics['tp'])} / 이상 판정 {fmt_int(predicted)}"),
                         value_color=score_color),
        make_metric_card("Recall · 이상 누락 방지", score(metrics["recall"]),
                         score_sub(f"TP {fmt_int(metrics['tp'])} / 실제 이상 {fmt_int(actual_anomaly)}"),
                         style={"border": f"1.5px solid {TEAL_500}"},
                         value_color=TEAL_600 if has_anomaly else GRAY),
        make_metric_card("F1", score(metrics["f1"]),
                         score_sub("Precision · Recall 조화평균"), value_color=score_color),
        make_metric_card("오경보율 (FPR)", f"{metrics['fpr']:.4f}",
                         f"FP {fmt_int(metrics['fp'])} / 실제 정상 {fmt_int(actual_normal)}"),
        make_metric_card("이벤트 탐지",
                         f"{int(metrics['event_detected'])} / {int(metrics['event_total'])}"
                         if metrics["event_total"] else NOT_APPLICABLE,
                         f"누락 {missed}건" if metrics["event_total"] else "이상 이벤트 없음",
                         value_color=INK if metrics["event_total"] else GRAY),
    ]


# ---------------------------------------------------------------- 혼동행렬

def make_confusion_matrix(metrics):
    if metrics is None:
        return make_empty_message()

    def cell(name, caption, value, style):
        return html.Div(
            children=[
                html.Div(f"{name} · {caption}", style={"fontSize": "12px", "opacity": "0.8"}),
                html.Div(fmt_int(value), style={"fontFamily": FONT_NUM, "fontSize": "26px",
                                                "fontWeight": "600", "marginTop": "4px"}),
            ],
            style={"borderRadius": "8px", "padding": "16px", "minHeight": "72px",
                   "boxSizing": "border-box", **style},
        )

    axis_text = {"fontSize": "11px", "color": GRAY}
    return html.Div(
        children=[
            html.Div(),
            html.Div("판정 정상", style={**axis_text, "textAlign": "center"}),
            html.Div("판정 이상", style={**axis_text, "textAlign": "center"}),

            html.Div("실제 정상", style={**axis_text, "alignSelf": "center"}),
            cell("TN", "정상 통과", metrics["tn"], {"backgroundColor": LIGHT, "color": INK}),
            cell("FP", "오경보", metrics["fp"], {"backgroundColor": TEAL_050, "color": INK}),

            html.Div("실제 이상", style={**axis_text, "alignSelf": "center"}),
            cell("FN", "누락", metrics["fn"], {"backgroundColor": WHITE, "color": INK,
                                              "border": f"2px solid {TEAL_500}"}),
            cell("TP", "이상 탐지", metrics["tp"], {"backgroundColor": INK, "color": WHITE}),
        ],
        style={"display": "grid", "gridTemplateColumns": "56px 1fr 1fr", "gap": "8px"},
    )


# ---------------------------------------------------------------- 모듈 요약 카드 4개

def make_module_summary(module_df):
    counts = data.judge_counts(module_df)
    total = len(module_df)
    good = counts.get(data.GOOD, 0)
    recheck = counts.get(data.RECHECK, 0)
    suspect = counts.get(data.SUSPECT, 0)

    recheck_by_file = (
        module_df[module_df["judgement"] == data.RECHECK]["file"].map(data.short_name).value_counts().sort_index()
    )
    recheck_text = " · ".join(f"{f} {n}" for f, n in recheck_by_file.items()) or "없음"

    suspect_df = module_df[module_df["judgement"] == data.SUSPECT].sort_values(["file", "cycle_id"])
    suspect_text = " ".join(
        f"{data.short_name(f)} #{int(c)}" if suspect_df["file"].nunique() > 1 else f"#{int(c)}"
        for f, c in zip(suspect_df["file"], suspect_df["cycle_id"])
    ) or "없음"
    if suspect_df["file"].nunique() == 1:
        suspect_text = f"{data.short_name(suspect_df['file'].iloc[0])} · {suspect_text}"

    good_ratio = f"이상 0 · {good / total * 100:.1f}%" if total else "이상 0"

    return [
        make_metric_card("전체 모듈", fmt_int(total), "1 모듈 = 39포인트"),
        make_metric_card("양품", fmt_int(good), good_ratio),
        make_metric_card("재검사 권고", fmt_int(recheck), f"이상 1–2 · {recheck_text}",
                         style={"borderTop": f"3px solid {TEAL_200}"}),
        make_metric_card("불량 의심", fmt_int(suspect), f"이상 3↑ 또는 0W · {suspect_text}",
                         style={"backgroundColor": TEAL_050, "border": f"1px solid {TEAL_500}"},
                         value_color=TEAL_600),
    ]


def make_chip_options(module_df):
    counts = data.judge_counts(module_df)
    options = [{"label": f"{data.ALL} {len(module_df)}", "value": data.ALL}]
    for judge in [data.GOOD, data.RECHECK, data.SUSPECT]:
        options.append({"label": f"{data.JUDGE_LABEL[judge]} {counts.get(judge, 0)}", "value": judge})
    return options


# ---------------------------------------------------------------- 모듈 목록 표

def badge_html(judgement):
    return f'<span class="badge {BADGE_CLASS.get(judgement, "")}">{data.JUDGE_LABEL.get(judgement, judgement)}</span>'


def page_map_html(anomaly_pages, zero_pages):
    cells = []
    for page in range(data.PAGE_MIN, data.PAGE_MAX + 1):
        if page in zero_pages:          # 미출력(0W)을 먼저 표시
            cls = "z"
        elif page in anomaly_pages:
            cls = "a"
        else:
            cls = "n"
        cells.append(f'<i class="{cls}" title="PageNo {page}"></i>')
    return f'<div class="pmap">{"".join(cells)}</div>'


def make_table_records(module_df, zero_pages_by_module):
    records = []
    for row in data.sort_modules(module_df).itertuples(index=False):
        anomaly_pages = data.parse_pages(row.anomaly_page_nos)
        zero_pages = zero_pages_by_module.get((row.file, row.cycle_id), set())
        records.append({
            "id": data.module_id(row.file, row.cycle_id),
            "judgement": row.judgement,
            "judge_badge": badge_html(row.judgement),
            "file_short": data.short_name(row.file),
            "module_label": f"#{int(row.cycle_id)}",
            "start_label": row.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "anomaly_label": f"{int(row.anomaly_points)}/{int(row.points)}",
            "zero_power_points": int(row.zero_power_points),
            "max_score_label": f"{row.max_anomaly_score:.2f}",
            "page_map": page_map_html(anomaly_pages, zero_pages),
        })
    return records


# ---------------------------------------------------------------- 선택 모듈 카드

def make_selected_module(module_row, zero_pages):
    if module_row is None:
        return [html.H3("선택 모듈", style={**CARD_TITLE_STYLE, "marginBottom": "12px"}),
                make_empty_message("목록에서 모듈을 선택하세요.")]

    anomaly_pages = data.parse_pages(module_row["anomaly_page_nos"])
    judgement = module_row["judgement"]

    cells = []
    for page in range(data.PAGE_MIN, data.PAGE_MAX + 1):
        if page in zero_pages:
            style = {"backgroundColor": INK, "color": WHITE}
        elif page in anomaly_pages:
            style = {"backgroundColor": TEAL_500, "color": INK, "fontWeight": "700"}
        else:
            style = {"backgroundColor": LIGHT, "color": GRAY}
        cells.append(html.Div(str(page), style={
            "height": "30px", "lineHeight": "30px", "textAlign": "center",
            "borderRadius": "4px", "fontFamily": FONT_NUM, "fontSize": "12px", **style,
        }))

    title = html.Div(
        children=[
            html.Span(data.JUDGE_LABEL[judgement], style={
                **BADGE_STYLE[judgement], "borderRadius": "4px", "padding": "2px 8px",
                "fontSize": "12px", "fontWeight": "700",
            }),
            html.H3(f"{data.short_name(module_row['file'])} · 모듈 #{int(module_row['cycle_id'])}",
                    style=CARD_TITLE_STYLE),
            html.Span(f"이상 {int(module_row['anomaly_points'])}/{int(module_row['points'])}",
                      style={"fontFamily": FONT_NUM, "fontSize": "12px", "color": GRAY,
                             "marginLeft": "auto"}),
        ],
        style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "14px"},
    )

    grid = html.Div(cells, style={"display": "grid", "gridTemplateColumns": "repeat(13, 1fr)",
                                  "gap": "4px"})
    return [title, grid]
