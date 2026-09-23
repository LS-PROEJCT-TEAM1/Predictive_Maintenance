# Guidebook Reproduction Model Comparison

This experiment reproduces the guidebook's Part 94 and Part 94+95 setups. Both setups predict Part 94 only.

## Guidebook LSTM comparison

| Scenario | Guidebook_Scaled_MAE | Our_Scaled_MAE | Guidebook_Inverse_MAE | Our_Inverse_MAE | Inverse_MAE_Difference |
| --- | --- | --- | --- | --- | --- |
| Part 94 only | 1.203700 | 1.181042 | 5.449800 | 5.347251 | -0.102549 |
| Part 94 + Part 95 | 1.087800 | 1.207740 | 4.925500 | 5.468129 | 0.542629 |

## Part 94 only

| MAE_Rank | Model | N | MAE | RMSE | WAPE_pct | Forecast_Bias | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LightGBM | 9 | 5.230099 | 5.880420 | 12.967187 | -0.467346 | 0.100538 |
| 2 | LSTM | 9 | 5.347251 | 6.190026 | 13.257647 | 1.140460 | 0.003330 |
| 3 | XGBoost | 9 | 5.541922 | 6.290271 | 13.740302 | 0.969149 | -0.029213 |
| 4 | CatBoost | 9 | 5.542222 | 6.218330 | 13.741047 | 0.633333 | -0.005805 |
| 5 | 3-day Moving Average | 9 | 8.111111 | 9.210166 | 20.110193 | 0.037037 | -1.206487 |

## Part 94 + Part 95

| MAE_Rank | Model | N | MAE | RMSE | WAPE_pct | Forecast_Bias | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LightGBM | 9 | 5.407447 | 6.190153 | 13.406895 | 0.662747 | 0.003289 |
| 2 | CatBoost | 9 | 5.459614 | 6.114026 | 13.536232 | 0.564271 | 0.027654 |
| 3 | LSTM | 9 | 5.468129 | 6.311226 | 13.557345 | 1.182587 | -0.036082 |
| 4 | XGBoost | 9 | 5.733362 | 6.559197 | 14.214946 | 1.302426 | -0.119097 |
| 5 | 3-day Moving Average | 9 | 8.111111 | 9.210166 | 20.110193 | 0.037037 | -1.206487 |

## Overall result

- Best scenario: Part 94 only
- Best model: LightGBM
- Best MAE: 5.230099
- Test size: 9 Part 94 target days
