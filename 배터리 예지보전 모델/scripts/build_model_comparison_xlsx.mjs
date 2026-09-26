import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const workspace = "C:\\Users\\USER\\Desktop\\배터리 예지보전 모델";
const csvPath = path.join(workspace, "outputs", "model_comparison", "model_comparison_metrics.csv");
const outputDir = path.join(workspace, "outputs", "model_comparison");
const outputPath = path.join(outputDir, "model_comparison_summary.xlsx");
const previewPath = path.join(outputDir, "model_comparison_summary_preview.png");

const csvText = await fs.readFile(csvPath, "utf8");
const lines = csvText.trim().split(/\r?\n/);
const headers = lines[0].split(",");
const rows = lines.slice(1).map((line) => {
  const values = line.split(",");
  return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
});

const number = (row, key) => Number(row[key]);
const modelMeta = {
  RobustZ: {
    type: "공정 단계별 통계 이상 탐지",
    temporal: "없음",
    verdict: "최종 선정 · 탐지 성능 공동 1위, 가장 단순·빠름",
  },
  LightGBM: {
    type: "정상 출력 회귀 기반 이상 탐지",
    temporal: "제한적(직전값)",
    verdict: "탐지 성능 공동 1위 · RobustZ보다 복잡·느림",
  },
  Cycle_AE: {
    type: "사이클 재구성 기반 이상 탐지",
    temporal: "사이클 내부 패턴",
    verdict: "이벤트는 모두 탐지했으나 행 단위 누락·오경보 증가",
  },
  EWMA_CUSUM: {
    type: "누적 변화 기반 순차 탐지",
    temporal: "있음(누적)",
    verdict: "Recall은 높지만 오경보가 가장 많음",
  },
  NHiTS_Z4: {
    type: "시계열 예측 오차 기반 이상 탐지",
    temporal: "있음(20→10 예측)",
    verdict: "시계열 활용 · 이벤트 1개 누락, 오경보 증가",
  },
};

const coreHeaders = [
  "순위", "모델", "모델 유형", "시계열 활용", "Precision", "Recall", "F1-score",
  "TP", "FP", "FN", "TN", "오경보율", "03·04 Macro F1",
];
const coreRows = rows.map((row) => [
  number(row, "rank"),
  row.model,
  modelMeta[row.model].type,
  modelMeta[row.model].temporal,
  number(row, "precision"),
  number(row, "recall"),
  number(row, "f1"),
  number(row, "tp"),
  number(row, "fp"),
  number(row, "fn"),
  number(row, "tn"),
  number(row, "fpr"),
  number(row, "macro_f1_03_04"),
]);

const opsHeaders = [
  "순위", "모델", "전체 이벤트", "이벤트 누락", "이벤트 Recall", "평균 탐지 지연(행)",
  "최대 탐지 지연(행)", "정상 파일 FP(행)", "정상 파일 오경보 이벤트", "정상 파일 오경보율",
  "학습 시간(초)", "추론 시간(ms/1,000행)", "모델 규모(파라미터)", "모델 크기(KB)", "평가",
];
const opsRows = rows.map((row) => [
  number(row, "rank"),
  row.model,
  number(row, "total_true_events"),
  number(row, "missed_events"),
  number(row, "event_recall"),
  number(row, "mean_detection_delay_rows"),
  number(row, "max_detection_delay_rows"),
  number(row, "normal_false_positive_rows"),
  number(row, "normal_false_alarm_events"),
  number(row, "normal_file_fpr"),
  number(row, "training_seconds"),
  number(row, "inference_ms_per_1000_rows"),
  number(row, "parameter_count"),
  number(row, "model_size_kb"),
  modelMeta[row.model].verdict,
]);

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("모델 비교");
sheet.showGridLines = false;
sheet.tabColor = "#17365D";

sheet.getRange("A2").values = [["사용 모델별 성능 비교표"]];
sheet.getRange("A2").format = {
  font: { name: "Arial", size: 16, bold: true, color: "#17365D" },
};
sheet.getRange("A3:O3").format.borders = {
  bottom: { style: "thin", color: "#9EADBA" },
};
sheet.getRange("A4").values = [["최종 선정: RobustZ — LightGBM과 탐지 성능은 동률이나 모델 복잡도와 추론 속도에서 우수"]];
sheet.getRange("A5").values = [["주의: 이 비교는 고장 발생 전 시점 예측이 아니라 라벨 기반 이상 탐지 성능 비교입니다."]];
sheet.getRange("A4:A5").format = {
  font: { name: "Arial", size: 10, italic: true, color: "#44546A" },
};

sheet.getRange("A7").values = [["1. 핵심 탐지 성능"]];
sheet.getRange("A7:M7").format = {
  fill: "#D9EAF7",
  font: { name: "Arial", size: 11, bold: true, color: "#17365D" },
  borders: { preset: "outside", style: "thin", color: "#9EADBA" },
};

sheet.getRange("A8:M13").values = [coreHeaders, ...coreRows];
const coreTable = sheet.tables.add("A8:M13", true, "CoreMetricsTable");
coreTable.style = "TableStyleMedium2";
coreTable.showBandedColumns = false;
coreTable.showFilterButton = true;

sheet.getRange("A16").values = [["2. 이벤트·운영 성능"]];
sheet.getRange("A16:O16").format = {
  fill: "#D9EAF7",
  font: { name: "Arial", size: 11, bold: true, color: "#17365D" },
  borders: { preset: "outside", style: "thin", color: "#9EADBA" },
};

sheet.getRange("A17:O22").values = [opsHeaders, ...opsRows];
const opsTable = sheet.tables.add("A17:O22", true, "OperationalMetricsTable");
opsTable.style = "TableStyleMedium2";
opsTable.showBandedColumns = false;
opsTable.showFilterButton = true;

sheet.getRange("A25").values = [["해석 메모"]];
sheet.getRange("A25:O25").format = {
  fill: "#E7E6E6",
  font: { name: "Arial", size: 10, bold: true, color: "#333333" },
  borders: { preset: "outside", style: "thin", color: "#BFBFBF" },
};
sheet.getRange("A26:A29").values = [
  ["• RobustZ와 LightGBM은 Precision, Recall, F1-score, 혼동행렬, 오경보율이 완전히 같습니다."],
  ["• 따라서 탐지 성능만으로는 동률이며, 최종 선정은 단순성·학습 시간·추론 속도·모델 크기로 결정했습니다."],
  ["• RobustZ는 시간 순서를 직접 학습하지 않으며, N-HiTS와 EWMA/CUSUM은 시간 흐름을 활용합니다."],
  ["• NHiTS_Z4는 현재 환경에서 재현한 경량 기준선이며 가이드북 수치와 1:1 동일 모델이라는 뜻은 아닙니다."],
];
sheet.getRange("A26:A29").format = {
  font: { name: "Arial", size: 10, color: "#44546A" },
};
sheet.getRange("A31").values = [["출처: outputs/model_comparison/model_comparison_metrics.csv"]];
sheet.getRange("A31").format = {
  font: { name: "Arial", size: 9, italic: true, color: "#7F7F7F" },
};

const populated = sheet.getRange("A2:O31");
populated.format.font.name = "Arial";
populated.format.verticalAlignment = "center";

sheet.getRange("A8:O8").format = {
  fill: "#17365D",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    insideVertical: { style: "thin", color: "#FFFFFF" },
    bottom: { style: "medium", color: "#17365D" },
  },
};
sheet.getRange("A17:O17").format = {
  fill: "#17365D",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    insideVertical: { style: "thin", color: "#FFFFFF" },
    bottom: { style: "medium", color: "#17365D" },
  },
};

sheet.getRange("A9:M13").format.rowHeight = 24;
sheet.getRange("A18:O22").format.rowHeight = 30;
sheet.getRange("A8:M8").format.rowHeight = 34;
sheet.getRange("A17:O17").format.rowHeight = 42;
sheet.getRange("A9:A13").format.horizontalAlignment = "center";
sheet.getRange("D9:D13").format.horizontalAlignment = "center";
sheet.getRange("A18:A22").format.horizontalAlignment = "center";

sheet.getRange("E9:G13").format.numberFormat = "0.00%";
sheet.getRange("L9:M13").format.numberFormat = "0.00%";
sheet.getRange("E18:E22").format.numberFormat = "0.00%";
sheet.getRange("J18:J22").format.numberFormat = "0.000%";
sheet.getRange("F18:G22").format.numberFormat = "0.00";
sheet.getRange("K18:L22").format.numberFormat = "0.000";
sheet.getRange("M18:M22").format.numberFormat = "#,##0";
sheet.getRange("N18:N22").format.numberFormat = "0.0";

sheet.getRange("A9:M9").format.fill = "#FFF2CC";
sheet.getRange("A18:O18").format.fill = "#FFF2CC";
sheet.getRange("B9:B9").format.font = { name: "Arial", bold: true, color: "#7F6000" };
sheet.getRange("B18:B18").format.font = { name: "Arial", bold: true, color: "#7F6000" };
sheet.getRange("E9:G13").conditionalFormats.add("colorScale", {
  colors: ["#F4CCCC", "#FFF2CC", "#D9EAD3"],
  thresholds: ["min", { type: "percentile", value: 50 }, "max"],
});
sheet.getRange("L9:L13").conditionalFormats.add("colorScale", {
  colors: ["#D9EAD3", "#FFF2CC", "#F4CCCC"],
  thresholds: ["min", { type: "percentile", value: 50 }, "max"],
});
sheet.getRange("M9:M13").conditionalFormats.add("colorScale", {
  colors: ["#F4CCCC", "#FFF2CC", "#D9EAD3"],
  thresholds: ["min", { type: "percentile", value: 50 }, "max"],
});

const widths = {
  A: 7, B: 16, C: 31, D: 20, E: 12, F: 12, G: 12, H: 9,
  I: 9, J: 9, K: 12, L: 14, M: 19, N: 16, O: 48,
};
for (const [column, width] of Object.entries(widths)) {
  sheet.getRange(`${column}:${column}`).format.columnWidth = width;
}
sheet.getRange("C9:D13").format.wrapText = true;
sheet.getRange("O18:O22").format.wrapText = true;
sheet.getRange("A26:O29").format.rowHeight = 21;

sheet.freezePanes.freezeRows(8);
sheet.freezePanes.freezeColumns(2);

workbook.recalculate();

const inspection = await workbook.inspect({
  kind: "region",
  sheetId: "모델 비교",
  range: "A2:O31",
  maxChars: 10000,
  tableMaxRows: 32,
  tableMaxCols: 15,
});
console.log(inspection.ndjson);

const preview = await workbook.render({
  sheetName: "모델 비교",
  range: "A1:O31",
  scale: 0.9,
  format: "png",
});
await fs.mkdir(outputDir, { recursive: true });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);

console.log(JSON.stringify({ outputPath, previewPath }));
