python EISFittingModel.py --mode=run_on_real_data --logdir=OnePercentTraining2
python EISFittingModel.py --mode=run_on_real_data --file_types=eis --logdir=OnePercentTraining2
python EISFittingModel.py --mode=finetune_test_data_with_adam --total_steps=1001 --logdir=OnePercentTraining2
python EISFittingModel.py --mode=finetune_test_data_with_adam --file_types=eis --total_steps=1001 --logdir=OnePercentTraining2

python PlotImpedance.py