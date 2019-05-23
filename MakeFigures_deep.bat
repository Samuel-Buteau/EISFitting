

python EISFittingModelDefinitions_v2.py --mode=run_inverse_model --logdir=OnePercentTraining_deep --num_conv=5 --conv_filters=20 --file_type=eis
python EISFittingModelDefinitions_v2.py --mode=run_inverse_model --logdir=OnePercentTraining_deep --num_conv=5 --conv_filters=20
python EISFittingModelDefinitions_v2.py --mode=finetune  --logdir=OnePercentTraining_hard --learning_rate=1e-1
python EISFittingModelDefinitions_v2.py --mode=finetune --file_types=eis  --logdir=OnePercentTraining_hard --learning_rate=1e-1

python PlotImpedance.py --output_dir=deep_3zarcs_inductance_zarc_inductance


python EISFittingModelDefinitions_v2.py --mode=run_inverse_model --logdir=OnePercentTraining_deep --num_conv=5 --conv_filters=20 --file_type=eis --num_zarcs=2
python EISFittingModelDefinitions_v2.py --mode=run_inverse_model --logdir=OnePercentTraining_deep --num_conv=5 --conv_filters=20 --num_zarcs=2
python EISFittingModelDefinitions_v2.py --mode=finetune  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=2
python EISFittingModelDefinitions_v2.py --mode=finetune --file_types=eis  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=2

python PlotImpedance.py --output_dir=deep_2zarcs_inductance_zarc_inductance

python EISFittingModelDefinitions_v2.py --mode=run_inverse_model --logdir=OnePercentTraining_deep --num_conv=5 --conv_filters=20 --file_type=eis --num_zarcs=1
python EISFittingModelDefinitions_v2.py --mode=run_inverse_model --logdir=OnePercentTraining_deep --num_conv=5 --conv_filters=20 --num_zarcs=1
python EISFittingModelDefinitions_v2.py --mode=finetune  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=1
python EISFittingModelDefinitions_v2.py --mode=finetune --file_types=eis  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=1

python PlotImpedance.py --output_dir=deep_1zarcs_inductance_zarc_inductance


REM

python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=run_inverse_model --logdir=OnePercentTraining_deep --num_conv=5 --conv_filters=20 --file_type=eis
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=run_inverse_model --logdir=OnePercentTraining_deep --num_conv=5 --conv_filters=20
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=finetune  --logdir=OnePercentTraining_hard --learning_rate=1e-1
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=finetune --file_types=eis  --logdir=OnePercentTraining_hard --learning_rate=1e-1

python PlotImpedance.py --output_dir=deep_3zarcs


python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=run_inverse_model --logdir=OnePercentTraining_deep --num_conv=5 --conv_filters=20 --file_type=eis --num_zarcs=2
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=run_inverse_model --logdir=OnePercentTraining_deep --num_conv=5 --conv_filters=20 --num_zarcs=2
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=finetune  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=2
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=finetune --file_types=eis  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=2

python PlotImpedance.py --output_dir=deep_2zarcs

python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=run_inverse_model --logdir=OnePercentTraining_deep --num_conv=5 --conv_filters=20 --file_type=eis --num_zarcs=1
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=run_inverse_model --logdir=OnePercentTraining_deep --num_conv=5 --conv_filters=20 --num_zarcs=1
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=finetune  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=1
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=finetune --file_types=eis  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=1

python PlotImpedance.py --output_dir=deep_1zarcs