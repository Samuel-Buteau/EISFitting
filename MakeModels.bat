REM python EISFittingModelDefinitions_v2.py --mode=train --logdir=OnePercentTraining_regression_test
REM python EISFittingModelDefinitions_v2.py --mode=train --logdir=OnePercentTraining_no_simplicity --num_zarcs_training_lower=1 --inductances_training_lower=0
REM python EISFittingModelDefinitions_v2.py --mode=train --logdir=OnePercentTraining_big --num_zarcs_training_lower=1 --inductances_training_lower=0 --num_conv=3 --conv_filters=24
REM python EISFittingModelDefinitions_v2.py --mode=train --logdir=OnePercentTraining_big_big --num_zarcs_training_lower=1 --inductances_training_lower=0 --num_conv=4 --conv_filters=32
REM python EISFittingModelDefinitions_v2.py --mode=train --logdir=OnePercentTraining_big3 --num_zarcs_training_lower=1 --inductances_training_lower=0 --num_conv=5 --conv_filters=40
REM python EISFittingModelDefinitions_v2.py --mode=train --logdir=OnePercentTraining_wide --num_zarcs_training_lower=1 --inductances_training_lower=0 --num_conv=3 --conv_filters=40
REM python EISFittingModelDefinitions_v2.py --mode=train --logdir=OnePercentTraining_deep --num_zarcs_training_lower=1 --inductances_training_lower=0 --num_conv=5 --conv_filters=20
REM python EISFittingModelDefinitions_v2.py --mode=train --logdir=OnePercentTraining_big4 --num_zarcs_training_lower=1 --inductances_training_lower=0 --num_conv=5 --conv_filters=56
python EISFittingModelDefinitions_v2.py --mode=train --logdir=OnePercentTraining_big3_new --num_zarcs_training_lower=1 --inductances_training_lower=0 --num_conv=5 --conv_filters=40
