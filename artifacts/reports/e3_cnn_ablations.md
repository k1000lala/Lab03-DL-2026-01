# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E3 | cnn_base | ninguno | COMPLETADO | 0.8746 | 0.8746 | 9.4320 | 13.2124 | 0.5412 | 286755 | 486.6781 |  |
| E3 | cnn_no_augmentation | sin aumentacion | COMPLETADO | 0.8766 | 0.8762 | 9.6720 | 13.5238 | 0.5194 | 286755 | 283.3288 |  |
| E3 | cnn_no_dropout | dropout=0.0 | COMPLETADO | 0.8755 | 0.8746 | 9.2008 | 12.7727 | 0.5713 | 286755 | 489.3536 |  |
| E3 | cnn_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8800 | 0.8798 | 13.3079 | 18.9578 | 0.0555 | 286755 | 486.9858 |  |
| E3 | cnn_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8485 | 0.8486 | 9.1748 | 12.8178 | 0.5682 | 286755 | 483.8293 |  |
