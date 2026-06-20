# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | classical_base | ninguno | COMPLETADO | 0.8440 | 0.8440 | 11.5972 | 14.9638 | 0.4115 | - | 14.2565 |  |
| E1 | classical_pca_50 | PCA=50 componentes | COMPLETADO | 0.8206 | 0.8206 | 12.1607 | 15.5764 | 0.3624 | - | 10.8191 |  |
| E1 | classical_pca_200 | PCA=200 componentes | COMPLETADO | 0.8561 | 0.8560 | 11.0047 | 14.3448 | 0.4592 | - | 20.6827 |  |
| E2 | mlp_base | ninguno | COMPLETADO | 0.8555 | 0.8556 | 12.1320 | 17.2238 | 0.2204 | 38568707 | 472.7996 |  |
| E2 | mlp_no_dropout | dropout=0.0 | COMPLETADO | 0.8563 | 0.8560 | 11.3739 | 16.0270 | 0.3250 | 38568707 | 478.9916 |  |
| E2 | mlp_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8434 | 0.8433 | 17.0716 | 23.6380 | -0.4684 | 38568707 | 488.3659 |  |
| E2 | mlp_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8319 | 0.8311 | 10.4366 | 14.8179 | 0.4230 | 38568707 | 480.8823 |  |
| E3 | cnn_base | ninguno | COMPLETADO | 0.8746 | 0.8746 | 9.4320 | 13.2124 | 0.5412 | 286755 | 486.6781 |  |
| E3 | cnn_no_augmentation | sin aumentacion | COMPLETADO | 0.8766 | 0.8762 | 9.6720 | 13.5238 | 0.5194 | 286755 | 283.3288 |  |
| E3 | cnn_no_dropout | dropout=0.0 | COMPLETADO | 0.8755 | 0.8746 | 9.2008 | 12.7727 | 0.5713 | 286755 | 489.3536 |  |
| E3 | cnn_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8800 | 0.8798 | 13.3079 | 18.9578 | 0.0555 | 286755 | 486.9858 |  |
| E3 | cnn_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8485 | 0.8486 | 9.1748 | 12.8178 | 0.5682 | 286755 | 483.8293 |  |
| E4 | resnet_frozen_base | ninguno | COMPLETADO | 0.8372 | 0.8371 | 10.5638 | 14.0479 | 0.4814 | 1539 | 493.7907 |  |
| E4 | resnet_frozen_no_augmentation | sin aumentacion | COMPLETADO | 0.8310 | 0.8307 | 10.5627 | 14.0809 | 0.4789 | 1539 | 296.6573 |  |
| E4 | resnet_frozen_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8372 | 0.8371 | 11.1793 | 15.0031 | 0.4085 | 1539 | 487.8068 |  |
| E4 | resnet_frozen_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8372 | 0.8371 | 10.5120 | 13.9555 | 0.4882 | 1539 | 499.2959 |  |
| E5 | resnet_finetuning_base | ninguno | COMPLETADO | 0.9202 | 0.9202 | 6.7546 | 9.5122 | 0.7622 | 8395267 | 506.1502 |  |
| E5 | resnet_finetuning_unfreeze_more | mas bloques descongelados | COMPLETADO | 0.9261 | 0.9261 | 7.5611 | 10.5955 | 0.7050 | 10494979 | 519.9885 |  |
| E5 | resnet_finetuning_lr_low | learning rate menor | COMPLETADO | 0.9157 | 0.9157 | 7.8900 | 11.1261 | 0.6747 | 8395267 | 506.9670 |  |
| E5 | resnet_finetuning_lambda_high | lambda_age=0.1 | COMPLETADO | 0.9277 | 0.9278 | 5.5617 | 7.9750 | 0.8329 | 8395267 | 506.0528 |  |
