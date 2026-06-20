# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E2 | mlp_base | ninguno | COMPLETADO | 0.8555 | 0.8556 | 12.1320 | 17.2238 | 0.2204 | 38568707 | 472.7996 |  |
| E2 | mlp_no_dropout | dropout=0.0 | COMPLETADO | 0.8563 | 0.8560 | 11.3739 | 16.0270 | 0.3250 | 38568707 | 478.9916 |  |
| E2 | mlp_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8434 | 0.8433 | 17.0716 | 23.6380 | -0.4684 | 38568707 | 488.3659 |  |
| E2 | mlp_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8319 | 0.8311 | 10.4366 | 14.8179 | 0.4230 | 38568707 | 480.8823 |  |
