# EISFitting
## Standard size for neural network
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--conv_filters', type=int, default=2*16)
    parser.add_argument('--dense_filters', type=int, default=16*16)

    parser.add_argument('--num_conv', type=int, default=8)
    parser.add_argument('--num_dense', type=int, default=2)

## Small size for neural network
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--conv_filters', type=int, default=1*16)
    parser.add_argument('--dense_filters', type=int, default=8*16)

    parser.add_argument('--num_conv', type=int, default=2)
    parser.add_argument('--num_dense', type=int, default=1)
    
## how to call the program to generate all figures. 

- First call ImpedanceRawFileExplorer
- then, call Impedance_fitting_model_v_20_2_2019.py --mode=train_on_all_data
- then, call Impedance_fitting_model_v_20_2_2019.py --mode=run_on_real_data 
- then, call Impedance_fitting_model_v_20_2_2019.py --mode=run_on_real_data --file_types=eis
- then, call Impedance_fitting_model_v_20_2_2019.py --mode=finetune_test_data_with_adam
- then, call Impedance_fitting_model_v_20_2_2019.py --mode=finetune_test_data_with_adam --file_types=eis

