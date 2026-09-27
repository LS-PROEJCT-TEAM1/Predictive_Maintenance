"""이상 탐지 탭 Callback 등록."""

from dash import Input, Output, State, ctx, dcc, no_update

from common.styles import INK, TEAL_050, TEAL_500
from tabs.anomaly import components, data, figures

# 임계값 표에 보여줄 분위수 (현재 적용값은 없으면 추가)
TABLE_QUANTILES = [0.99, 0.996, 0.999, 0.9999]


def make_threshold_table(sweep_df, applied_quantile):
    quantiles = set(TABLE_QUANTILES) | {applied_quantile}
    table_df = sweep_df[sweep_df["calibration_quantile"].isin(quantiles)]
    records = []
    for row in table_df.itertuples(index=False):
        mark = " ✓" if row.calibration_quantile == applied_quantile else ""
        records.append({
            "quantile": row.calibration_quantile,
            "quantile_label": f"{row.calibration_quantile * 100:g}%{mark}",
            "threshold_label": f"{row.threshold:.2f}",
            "fp": int(row.fp),
            "normal_false_alarm_events": int(row.normal_false_alarm_events),
        })
    style = [{
        "if": {"filter_query": f"{{quantile}} = {applied_quantile}"},
        "backgroundColor": TEAL_050, "fontWeight": "700",
    }]
    return records, style


def register_callbacks(app):

    # ------------------------------------------------ 임계값 슬라이더: [적용] [기본값]
    @app.callback(
        Output("anom-threshold-store", "data"),
        Output("anom-threshold-slider", "value"),
        Input("anom-apply-btn", "n_clicks"),
        Input("anom-reset-btn", "n_clicks"),
        State("anom-threshold-slider", "value"),
        prevent_initial_call=True,
    )
    def apply_threshold(apply_clicks, reset_clicks, slider_index):
        sweep = data.get_threshold_sweep()
        if ctx.triggered_id == "anom-reset-btn":
            default_q = data.get_default_quantile()
            default_index = int(sweep.index[sweep["calibration_quantile"] == default_q][0])
            return default_q, default_index
        return float(sweep.loc[slider_index, "calibration_quantile"]), no_update

    @app.callback(
        Output("anom-threshold-readout", "children"),
        Input("anom-threshold-slider", "value"),
        Input("anom-threshold-store", "data"),
    )
    def update_threshold_readout(slider_index, applied_quantile):
        row = data.get_threshold_sweep().loc[slider_index]
        text = f"{row['calibration_quantile'] * 100:g}% = {row['threshold']:.2f}"
        if row["calibration_quantile"] != applied_quantile:
            text += " (미적용)"
        return text

    # ------------------------------------------------ 상단 파일·기간 → 모듈 필터에 반영
    @app.callback(
        Output("anom-module-file", "value"),
        Output("anom-module-date", "start_date"),
        Output("anom-module-date", "end_date"),
        Input("anom-file", "value"),
        Input("anom-date", "start_date"),
        Input("anom-date", "end_date"),
    )
    def sync_module_filters(selected_file, start_date, end_date):
        return selected_file, start_date, end_date

    # ------------------------------------------------ 판정 칩: '전체'와 개별 판정을 함께 고르지 않게 정리
    @app.callback(
        Output("anom-judge-chips", "value"),
        Output("anom-judge-prev", "data"),
        Input("anom-judge-chips", "value"),
        State("anom-judge-prev", "data"),
        prevent_initial_call=True,
    )
    def normalize_chips(selected, previous):
        selected = selected or []
        added = set(selected) - set(previous or [])
        judges = [j for j in selected if j != data.ALL]

        if data.ALL in added or not selected or len(judges) == 3:
            result = [data.ALL]
        else:
            result = judges
        return result, result

    # ------------------------------------------------ 필터 → 카드·그래프·표 동시 갱신
    @app.callback(
        Output("anom-metrics", "children"),
        Output("anom-metric-note", "children"),
        Output("anom-confusion", "children"),
        Output("anom-judge-chips", "options"),
        Output("anom-module-summary", "children"),
        Output("anom-module-table", "data"),
        Output("anom-condition-graph", "figure"),
        Output("anom-threshold-graph", "figure"),
        Output("anom-threshold-table", "data"),
        Output("anom-threshold-table", "style_data_conditional"),
        Input("anom-file", "value"),
        Input("anom-date", "start_date"),
        Input("anom-date", "end_date"),
        Input("anom-setpower", "value"),
        Input("anom-pageno", "value"),
        Input("anom-threshold-store", "data"),
        Input("anom-judge-chips", "value"),
        Input("anom-module-file", "value"),
        Input("anom-module-date", "start_date"),
        Input("anom-module-date", "end_date"),
        Input("anom-module-search", "value"),
    )
    def update_anomaly_tab(
        selected_file, start_date, end_date, selected_setpower, page_range,
        applied_quantile, selected_judges,
        module_file, module_start, module_end, module_search,
    ):
        # 1) 행 단위 판정 결과 (predictions)
        pred_df = data.get_predictions()
        filtered_pred = data.filter_predictions(
            pred_df, selected_file, start_date, end_date, selected_setpower, page_range,
        )
        row_filtered = data.is_row_filtered(start_date, end_date, selected_setpower, page_range)

        metrics, note = data.get_metrics(filtered_pred, selected_file, applied_quantile, row_filtered)
        metric_cards = components.make_metric_cards(metrics)
        confusion = components.make_confusion_matrix(metrics)

        # 2) 모듈 판정 (module_judgement)
        module_df = data.filter_modules(
            data.get_modules(), module_file, module_start, module_end, module_search,
        )
        chip_options = components.make_chip_options(module_df)
        summary_cards = components.make_module_summary(module_df)
        table_records = components.make_table_records(
            data.filter_judges(module_df, selected_judges),
            data.zero_power_pages(pred_df),
        )

        # 3) 그래프
        if filtered_pred.empty:
            condition_fig = figures.make_empty_figure()
        else:
            unseen = set(pred_df.loc[pred_df["unseen_setpower"] == 1, "SetPower"])
            condition_fig = figures.make_condition_bar(data.condition_ratio(filtered_pred), unseen)

        sweep_df = data.get_threshold_sweep()
        threshold_fig = figures.make_threshold_curve(sweep_df, applied_quantile)
        threshold_records, threshold_style = make_threshold_table(sweep_df, applied_quantile)

        return (
            metric_cards, note, confusion,
            chip_options, summary_cards, table_records,
            condition_fig, threshold_fig, threshold_records, threshold_style,
        )

    # ------------------------------------------------ 표 행 클릭 → 선택 모듈 카드
    @app.callback(
        Output("anom-selected-module", "children"),
        Output("anom-module-table", "style_data_conditional"),
        Input("anom-module-table", "active_cell"),
        Input("anom-module-table", "data"),
    )
    def update_selected_module(active_cell, table_records):
        if not table_records:
            return components.make_selected_module(None, set()), []

        record_ids = [r["id"] for r in table_records]
        selected_id = active_cell.get("row_id") if active_cell else None
        if selected_id not in record_ids:
            selected_id = record_ids[0]

        file_name, cycle_id = selected_id.split("|")
        modules = data.get_modules()
        module_row = modules[(modules["file"] == file_name) & (modules["cycle_id"] == int(cycle_id))].iloc[0]
        zero_pages = data.zero_power_pages(data.get_predictions()).get((file_name, int(cycle_id)), set())

        highlight = [
            {"if": {"filter_query": f'{{id}} = "{selected_id}"'},
             "backgroundColor": TEAL_050},
            {"if": {"state": "active"},
             "backgroundColor": TEAL_050, "border": f"1px solid {TEAL_500}", "color": INK},
        ]
        return components.make_selected_module(module_row, zero_pages), highlight

    # ------------------------------------------------ 모듈 목록 다운로드 (현재 필터 기준)
    @app.callback(
        Output("anom-download", "data"),
        Input("anom-download-btn", "n_clicks"),
        State("anom-judge-chips", "value"),
        State("anom-module-file", "value"),
        State("anom-module-date", "start_date"),
        State("anom-module-date", "end_date"),
        State("anom-module-search", "value"),
        prevent_initial_call=True,
    )
    def download_modules(n_clicks, selected_judges, module_file, module_start, module_end, module_search):
        module_df = data.filter_modules(
            data.get_modules(), module_file, module_start, module_end, module_search,
        )
        module_df = data.sort_modules(data.filter_judges(module_df, selected_judges))
        csv_bytes = module_df.to_csv(index=False).encode("utf-8-sig")
        return dcc.send_bytes(csv_bytes, "module_judgement_filtered.csv")
