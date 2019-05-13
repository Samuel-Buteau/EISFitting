python EISFittingModelDefinitions.py --mode=run_inverse_model --logdir=OnePercentTraining --file_type=eis
python EISFittingModelDefinitions.py --mode=run_inverse_model --logdir=OnePercentTraining
python EISFittingModelDefinitions.py --mode=finetune  --logdir=OnePercentTraining --learning_rate=1e-1
python EISFittingModelDefinitions.py --mode=finetune --file_types=eis  --logdir=OnePercentTraining --learning_rate=1e-1

python PlotImpedance.py