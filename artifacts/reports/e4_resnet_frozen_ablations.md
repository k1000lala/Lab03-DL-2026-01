# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E4 | resnet_frozen_base | ninguno | COMPLETADO | 0.8372 | 0.8371 | 10.5638 | 14.0479 | 0.4814 | 1539 | 493.7907 |  |
| E4 | resnet_frozen_no_augmentation | sin aumentacion | COMPLETADO | 0.8310 | 0.8307 | 10.5627 | 14.0809 | 0.4789 | 1539 | 296.6573 |  |
| E4 | resnet_frozen_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8372 | 0.8371 | 11.1793 | 15.0031 | 0.4085 | 1539 | 487.8068 |  |
| E4 | resnet_frozen_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8372 | 0.8371 | 10.5120 | 13.9555 | 0.4882 | 1539 | 499.2959 |  |
