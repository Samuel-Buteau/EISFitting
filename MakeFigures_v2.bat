

python EISFittingModelDefinitions_v2.py --mode=run_inverse_model --logdir=OnePercentTraining_big_big --num_conv=4 --conv_filters=32 --file_type=eis
python EISFittingModelDefinitions_v2.py --mode=run_inverse_model --logdir=OnePercentTraining_big_big --num_conv=4 --conv_filters=32
python EISFittingModelDefinitions_v2.py --mode=finetune  --logdir=OnePercentTraining_hard --learning_rate=1e-1
python EISFittingModelDefinitions_v2.py --mode=finetune --file_types=eis  --logdir=OnePercentTraining_hard --learning_rate=1e-1

python PlotImpedance.py --output_dir=Big_Big_3zarcs_inductance_zarc_inductance


python EISFittingModelDefinitions_v2.py --mode=run_inverse_model --logdir=OnePercentTraining_big_big --num_conv=4 --conv_filters=32 --file_type=eis --num_zarcs=2
python EISFittingModelDefinitions_v2.py --mode=run_inverse_model --logdir=OnePercentTraining_big_big --num_conv=4 --conv_filters=32 --num_zarcs=2
python EISFittingModelDefinitions_v2.py --mode=finetune  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=2
python EISFittingModelDefinitions_v2.py --mode=finetune --file_types=eis  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=2

python PlotImpedance.py --output_dir=Big_Big_2zarcs_inductance_zarc_inductance

python EISFittingModelDefinitions_v2.py --mode=run_inverse_model --logdir=OnePercentTraining_big_big --num_conv=4 --conv_filters=32 --file_type=eis --num_zarcs=1
python EISFittingModelDefinitions_v2.py --mode=run_inverse_model --logdir=OnePercentTraining_big_big --num_conv=4 --conv_filters=32 --num_zarcs=1
python EISFittingModelDefinitions_v2.py --mode=finetune  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=1
python EISFittingModelDefinitions_v2.py --mode=finetune --file_types=eis  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=1

python PlotImpedance.py --output_dir=Big_Big_1zarcs_inductance_zarc_inductance


REM

python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=run_inverse_model --logdir=OnePercentTraining_big_big --num_conv=4 --conv_filters=32 --file_type=eis
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=run_inverse_model --logdir=OnePercentTraining_big_big --num_conv=4 --conv_filters=32
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=finetune  --logdir=OnePercentTraining_hard --learning_rate=1e-1
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=finetune --file_types=eis  --logdir=OnePercentTraining_hard --learning_rate=1e-1

python PlotImpedance.py --output_dir=Big_Big_3zarcs


python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=run_inverse_model --logdir=OnePercentTraining_big_big --num_conv=4 --conv_filters=32 --file_type=eis --num_zarcs=2
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=run_inverse_model --logdir=OnePercentTraining_big_big --num_conv=4 --conv_filters=32 --num_zarcs=2
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=finetune  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=2
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=finetune --file_types=eis  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=2

python PlotImpedance.py --output_dir=Big_Big_2zarcs

python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=run_inverse_model --logdir=OnePercentTraining_big_big --num_conv=4 --conv_filters=32 --file_type=eis --num_zarcs=1
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=run_inverse_model --logdir=OnePercentTraining_big_big --num_conv=4 --conv_filters=32 --num_zarcs=1
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=finetune  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=1
python EISFittingModelDefinitions_v2.py --no-zarc-inductance --no-inductance --mode=finetune --file_types=eis  --logdir=OnePercentTraining_hard --learning_rate=1e-1 --num_zarcs=1

python PlotImpedance.py --output_dir=Big_Big_1zarcs