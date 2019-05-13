python EISFittingModel_reg_conv_no_synth_masked_v2.py --mode=run_on_real_data --logdir=OnePercentTraining_newest_4hours_tight_crop --file_type=eis
python EISFittingModel_reg_conv_no_synth_masked_v2.py --mode=run_on_real_data --logdir=OnePercentTraining_newest_4hours_tight_crop
python EISFittingModel_reg_conv_no_synth_masked_v2.py --mode=finetune_test_data_with_adam  --logdir=OnePercentTraining_newest_4hours_tight_crop --learning_rate=1e-1
python EISFittingModel_reg_conv_no_synth_masked_v2.py --mode=finetune_test_data_with_adam --file_types=eis  --logdir=OnePercentTraining_newest_4hours_tight_crop --learning_rate=1e-1

python PlotImpedance.py