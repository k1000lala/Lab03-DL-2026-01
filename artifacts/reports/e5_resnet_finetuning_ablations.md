# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E5 | resnet_finetuning_base | ninguno | COMPLETADO | 0.9202 | 0.9202 | 6.7546 | 9.5122 | 0.7622 | 8395267 | 506.1502 |  |
| E5 | resnet_finetuning_unfreeze_more | mas bloques descongelados | COMPLETADO | 0.9261 | 0.9261 | 7.5611 | 10.5955 | 0.7050 | 10494979 | 519.9885 |  |
| E5 | resnet_finetuning_lr_low | learning rate menor | COMPLETADO | 0.9157 | 0.9157 | 7.8900 | 11.1261 | 0.6747 | 8395267 | 506.9670 |  |
| E5 | resnet_finetuning_lambda_high | lambda_age=0.1 | COMPLETADO | 0.9277 | 0.9278 | 5.5617 | 7.9750 | 0.8329 | 8395267 | 506.0528 |  |
