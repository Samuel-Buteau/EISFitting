import contextlib
import csv
import math
import os
import pickle
import random




import copy


import matplotlib


import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from django.core.management.base import BaseCommand
import re

from import_eis_files import import_eis_file
from EIS.models import *

from matplotlib.gridspec import GridSpec




from EISFittingModelDefinitions import Prior,ParameterVAE, shift_scale_param_extract,normalized_spectrum,original_spectrum, initialize_session,NonparametricOptimizer, run_through_trained_model, run_optimizer_on_data






def get_inv_model_results_or_none(spectrum, inv_model, inductance, zarc_inductance, num_zarcs):
    '''

    returns two values. first is the inv_model_results. second is a bool saying if it existed.
    :param spectrum:
    :param inv_model:
    :return:

    '''
    query = InverseModelResult.objects.filter(spectrum=spectrum, inv_model=inv_model, inductance=inductance, zarc_inductance=zarc_inductance, num_zarcs=num_zarcs)
    if query.exists():
        for inv_model_res in query.all():
            match = True
            for activity_sample in ActivitySample.objects.filter(setting=inv_model_res.activity_setting):
                if not activity_sample.active == activity_sample.sample.active:
                    match = False
                    break

            if match == True:
                return inv_model_res, True

    return None, False

def get_finetune_results_or_none(inv_model_result, args):
    '''

    returns two values. first is the finetune_results. second is a bool saying if it existed.
    :param spectrum:
    :param inv_model:
    :return:

    '''
    query = FinetuneResult.objects.filter(
            inv_model_result=inv_model_result,
            learning_rate__range=(args['learning_rate']*.9,args['learning_rate']*1.1, ),
            nll_coeff__range=(args['nll_coeff']-0.01, args['nll_coeff']+0.01),
            ordering_coeff__range=(args['ordering_coeff'] - 0.01, args['ordering_coeff'] + 0.01),
            simplicity_coeff__range=(args['simplicity_coeff'] - 0.01, args['simplicity_coeff'] + 0.01),
            sensible_phi_coeff__range=(args['sensible_phi_coeff'] - 0.01, args['sensible_phi_coeff'] + 0.01),

    )
    if query.exists():
        return query[0], True

    return None, False







def import_process_output(args):
    number_of_zarcs = 3
    number_of_params = 1 + 1 + number_of_zarcs + 1 + 1 + 1 + number_of_zarcs + 1 + number_of_zarcs + 1 + 1
    print('starting up...')
    if not os.path.exists(args['input_dir']):
        print('Please provide a valid value for --input_dir')
        return

    user_dataset, _ = Dataset.objects.get_or_create(label=args['dataset'])

    all_filenames = []
    path_to_robot = args['input_dir']
    for root, dirs, filenames in os.walk(path_to_robot):
        for file in filenames:
            if file.endswith('.mpt'):
                all_filenames.append(os.path.join(root, file))

    # for now, don't record in database, since it takes a long time.
    all_spectra = []
    all_samples = []

    for filename in all_filenames:
        with open(filename, 'rb') as f:
            record, valid = import_eis_file(f)

        if valid:

            if  EISSpectrum.objects.filter(filename=str(filename), dataset__label=args['dataset']).exists():
                continue

            aas = AutomaticActiveSample(
                freqs_with_tails_im_z=record['freqs_with_tails_im_z'],
                freqs_with_negative_im_z=record['freqs_with_negative_im_z'],
                sample_count=len(record['original_spectrum'][0])
            )
            aas.save()



            new_spectrum = EISSpectrum(filename=str(filename),
                                       dataset=user_dataset,
                                       automatic_active_sample=aas,
                                       )
            new_spectrum.save()
            all_spectra.append(new_spectrum)

            for index in range(aas.sample_count):
                if index < aas.sample_to_keep_count:
                    active = True
                else:
                    active = False
                new_sample = ImpedanceSample(
                    spectrum=new_spectrum,
                    log_ang_freq=record['original_spectrum'][0][index],
                    real_part=record['original_spectrum'][1][index],
                    imag_part=record['original_spectrum'][2][index],
                    active=active
                )
                all_samples.append(new_sample)

    ImpedanceSample.objects.bulk_create(all_samples)

    #process 1
    print('running neural net')

    if not InverseModel.objects.filter(logdir=args['logdir']).exists():
        raise Exception('Parameter --logdir does not correspond to a valid model.')
    inv_model = InverseModel.objects.get(logdir=args['logdir'])
    print('Now using {}.'.format(inv_model.display()))







    cleaned_data = []

    activity_sample_list = []
    for spectrum in user_dataset.eisspectrum_set.filter(active=True):
        if not spectrum.any_active():
            continue

        _, already_exists = get_inv_model_results_or_none(
            spectrum, inv_model,
            inductance=args['inductance'],
            zarc_inductance=args['zarc_inductance'],
            num_zarcs=args['num_zarcs'])
        if already_exists:
            continue



        spec_numpy = spectrum.get_sample_array()

        spec_tuple = (spec_numpy[:, 0], spec_numpy[:, 1], spec_numpy[:, 2])
        shift_scale_params = shift_scale_param_extract(spec_tuple)

        ssp = ShiftScaleParameters(
            r_alpha=shift_scale_params['r_alpha'],
            w_alpha=shift_scale_params['w_alpha'],
        )
        ssp.save()

        cp = CircuitParameterSet(
            circuit='standard3zarc'
        )
        cp.save()

        fs = FitSpectrum()
        fs.save()

        activity_setting = ActivitySetting()
        activity_setting.save()

        inverse_model_results = InverseModelResult(
            spectrum=spectrum,
            inv_model=inv_model,
            inductance=args['inductance'],
            zarc_inductance=args['zarc_inductance'],
            num_zarcs=args['num_zarcs'],
            activity_setting=activity_setting,
            shift_scale_parameters=ssp,
            circuit_parameters=cp,
            fit_spectrum=fs,
        )
        inverse_model_results.save()

        for samp in spectrum.impedancesample_set.filter(active=True):
            activity_sample_list.append(
                ActivitySample(
                    setting=activity_setting,
                    sample=samp,
                    active=samp.active,
                )
            )

        log_freq, re_z, im_z = normalized_spectrum(spec_tuple,
                                                   params=inverse_model_results.shift_scale_parameters.to_dict())

        cleaned_data.append((log_freq, re_z, im_z, inverse_model_results.id))

    ActivitySample.objects.bulk_create(activity_sample_list)

    results = run_through_trained_model(
        cleaned_data=cleaned_data,
        inverse_model_params={
            'kernel_size': inv_model.kernel_size,
            'num_conv': inv_model.num_conv,
            'conv_filters': inv_model.conv_filters,
            'logdir': inv_model.logdir,
            'inductance' : args['inductance'],
            'zarc_inductance' : args['zarc_inductance'],
            'num_zarcs' : args['num_zarcs'],
        },
        seed=args['seed'],
        chunk_num=args['chunk_num']
    )



    if len(results) > 0:
        circuit_parameters_list = []
        fit_samples_list = []
        for freqs, in_impedance, out_impedance, representation_mu_value, id in results:
            inv_model_result = InverseModelResult.objects.get(id=id)

            for i in range(len(representation_mu_value)):
                circuit_parameters_list.append(
                    CircuitParameter(
                        set=inv_model_result.circuit_parameters,
                        index=i,
                        value=representation_mu_value[i]
                    )
                )

            for i in range(len(freqs)):
                fit_samples_list.append(
                    FitSample(
                        fit=inv_model_result.fit_spectrum,
                        log_ang_freq=freqs[i],
                        real_part=out_impedance[i, 0],
                        imag_part=out_impedance[i, 1],
                    )
                )
        CircuitParameter.objects.bulk_create(circuit_parameters_list)
        FitSample.objects.bulk_create(fit_samples_list)

   # process 2
    print('running finetuning')


    cleaned_data = []
    for spectrum in user_dataset.eisspectrum_set.filter(active=True):
        if not spectrum.any_active():
            continue

        inv_model_result, already_exists = get_inv_model_results_or_none(
            spectrum, inv_model,
            inductance=args['inductance'],
            zarc_inductance=args['zarc_inductance'],
            num_zarcs=args['num_zarcs']
        )


        assert already_exists

        _, already_exists = get_finetune_results_or_none(inv_model_result, args)
        if not already_exists:
            cp = CircuitParameterSet(
                circuit='standard3zarc'
            )
            cp.save()

            fs = FitSpectrum()
            fs.save()

            finetune_results = FinetuneResult(
                inv_model_result=inv_model_result,
                learning_rate=args['learning_rate'],
                nll_coeff=args['nll_coeff'],
                ordering_coeff=args['ordering_coeff'],
                simplicity_coeff=args['simplicity_coeff'],
                sensible_phi_coeff=args['sensible_phi_coeff'],

                circuit_parameters=cp,
                fit_spectrum=fs,
            )

            finetune_results.save()

            norm_arr = inv_model_result.get_normalized_sample_array()

            cleaned_data.append(
                (
                    norm_arr[:,0],
                    norm_arr[:,1:],
                    norm_arr[:, 1:],
                    inv_model_result.circuit_parameters.get_parameter_array(),
                    finetune_results.id
                )
            )


    results = run_optimizer_on_data(
        cleaned_data=cleaned_data,
        args={
            'learning_rate': args['learning_rate'],
            'sensible_phi_coeff': args['sensible_phi_coeff'],
            'simplicity_coeff': args['simplicity_coeff'],
            'nll_coeff': args['nll_coeff'],
            'ordering_coeff': args['ordering_coeff'],
            'inductance': args['inductance'],
            'zarc_inductance': args['zarc_inductance'],
            'num_zarcs': args['num_zarcs'],

        },
        chunk_num=args['chunk_num'] * 32
    )



    circuit_parameters_list = []
    fit_samples_list = []

    for freqs, in_impedance, out_impedance, params, id in results:

        finetune_results = FinetuneResult.objects.get(id=id)

        for i in range(len(params)):
            circuit_parameters_list.append(
                CircuitParameter(
                    set=finetune_results.circuit_parameters,
                    index=i,
                    value=params[i]
                )
            )

        for i in range(len(freqs)):
            fit_samples_list.append(
                FitSample(
                    fit=finetune_results.fit_spectrum,
                    log_ang_freq=freqs[i],
                    real_part=out_impedance[i, 0],
                    imag_part=out_impedance[i, 1],
                )
            )
    CircuitParameter.objects.bulk_create(circuit_parameters_list)
    FitSample.objects.bulk_create(fit_samples_list)

    # this outputs the results.
    print('outputing the results')
    if not os.path.exists(args['output_dir']):
        os.mkdir(args['output_dir'])

    results = []
    for spectrum in user_dataset.eisspectrum_set.filter(active=True):
        if not spectrum.any_active():
            continue

        inv_model_result, should_be_true = get_inv_model_results_or_none(
            spectrum, inv_model,
            inductance=args['inductance'],
            zarc_inductance=args['zarc_inductance'],
            num_zarcs=args['num_zarcs']
        )

        assert should_be_true

        
        finetune_results, already_exists = get_finetune_results_or_none(inv_model_result, args)
        assert already_exists

        orig_spec =  spectrum.get_sample_array()


        fit_spec_unnorm = inv_model_result.fit_spectrum.get_sample_array()
        freq_fit, real_fit, imag_fit = original_spectrum(
            (fit_spec_unnorm[:, 0], fit_spec_unnorm[:, 1], fit_spec_unnorm[:, 2],),
            inv_model_result.shift_scale_parameters.to_dict())
        fit_spec = numpy.stack((freq_fit, real_fit, imag_fit), axis=-1)


        fit_spec_unnorm2 = finetune_results.fit_spectrum.get_sample_array()
        freq_fit2, real_fit2, imag_fit2 = original_spectrum(
            (fit_spec_unnorm2[:, 0], fit_spec_unnorm2[:, 1], fit_spec_unnorm2[:, 2],),
            inv_model_result.shift_scale_parameters.to_dict())
        fit_spec2 = numpy.stack((freq_fit2, real_fit2, imag_fit2), axis=-1)


        if args['angular_freq']:

            orig_spec[:, 0] = numpy.exp(orig_spec[:, 0])
            fit_spec[:, 0]  =  numpy.exp(fit_spec[:, 0])
            fit_spec2[:, 0] = numpy.exp(fit_spec2[:, 0])
            freq_units = '(rad/s)'
            freq_symbol = 'Angular Freq'
        else:

            orig_spec[:, 0] = 1. / (2. * numpy.pi) * numpy.exp(orig_spec[:, 0])
            fit_spec[:, 0] = 1. / (2. * numpy.pi) * numpy.exp(fit_spec[:, 0])
            fit_spec2[:, 0] = 1. / (2. * numpy.pi) * numpy.exp(fit_spec2[:, 0])
            freq_units = '(Hz)'
            freq_symbol = 'Freq'



        filename_output = spectrum.filename
        filename_output = filename_output.split('.mpt')[0].replace('\\', '__').replace('/', '__')
        fig = plt.figure()
        gs = GridSpec(2, 2, figure=fig)

        ax = fig.add_subplot(gs[0, :])
        ax.scatter(orig_spec[:, 1], -orig_spec[:, 2], c='k', label='Original Data')
        ax.plot(fit_spec[:, 1], -fit_spec[:, 2], c='r', label='Inverse Model Fit')
        ax.plot(fit_spec2[:, 1], -fit_spec2[:, 2], c='b', label='Finetuned Fit')
        ax.legend()

        ax.set_xlabel('Re[Z] (ohm)')
        ax.set_ylabel('-Im[Z] (ohm)')

        ax = fig.add_subplot(gs[1, 0])
        ax.set_xscale('log')
        ax.scatter(orig_spec[:, 0], -orig_spec[:, 2], c='k')
        ax.plot(fit_spec[:, 0], -fit_spec[:, 2], c='r')
        ax.plot(fit_spec2[:, 0], -fit_spec2[:, 2], c='b')
        ax.set_xlabel('{} {}'.format(freq_symbol, freq_units))
        ax.set_ylabel('Re[Z] (ohm)')

        ax = fig.add_subplot(gs[1, 1])
        ax.set_xscale('log')
        ax.scatter(orig_spec[:, 0], orig_spec[:, 1], c='k')
        ax.plot(fit_spec[:, 0], fit_spec[:, 1], c='r')
        ax.plot(fit_spec2[:, 0], fit_spec2[:, 1], c='b')
        ax.set_xlabel('{} {}'.format(freq_symbol, freq_units))
        ax.set_ylabel('-Im[Z] (ohm)')

        fig.tight_layout(h_pad=0., w_pad=0.)

        fig.savefig(os.path.join(args['output_dir'], filename_output + '_FIT_QUALITY_CONTROL.png'))
        plt.close(fig)

        with open(os.path.join(args['output_dir'], filename_output + '_FIT_QUALITY_CONTROL.csv'), 'w',
                  newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow([
                'Original Data {} {}'.format(freq_symbol,freq_units),
                'Original Data Re[Z] (ohm)',
                'Original Data Im[Z] (ohm)',
                'Inverse Model Fit {} {}'.format(freq_symbol,freq_units),
                'Inverse Model Fit Re[Z] (ohm)',
                'Inverse Model Fit Im[Z] (ohm)',
                'Finetuned Fit {} {}'.format(freq_symbol,freq_units),
                'Finetuned Fit Re[Z] (ohm)',
                'Finetuned Fit Im[Z] (ohm)',
            ])

            for k in range(len(orig_spec)):
                spamwriter.writerow([str(x) for x in [
                    orig_spec[k,0],
                    orig_spec[k,1],
                    orig_spec[k, 2],
                    fit_spec[k, 0],
                    fit_spec[k, 1],
                    fit_spec[k, 2],
                    fit_spec2[k, 0],
                    fit_spec2[k, 1],
                    fit_spec2[k, 2],
                ]])

        inv_p = inv_model_result.get_circuit_parameters_in_original_form()
        fine_p =finetune_results.get_circuit_parameters_in_original_form()
        delta_p = numpy.abs(inv_p-fine_p)
        results.append({
            'filename':spectrum.filename,
            'inverse_model_params':inv_p,
            'finetuned_params':fine_p,
            'delta_params':delta_p
        })

    #parameters

    all_labels = \
     [
         (True, '{} R (ohm)'),
         (args['zarc_impedance'], '{} R_zarc_impedance (ohm)'),
         (args['num_zarcs']>=1, '{} R_zarc_1 (ohm)'),
         (args['num_zarcs']>=2, '{} R_zarc_2 (ohm)'),
         (args['num_zarcs']>=3, '{} R_zarc_3 (ohm)'),
         (True, '{} Q_warburg (?)'),
         (args['inductance'], '{} Q_inductance (?)'),
         (args['zarc_impedance'], '{} W_c_inductance (rad/s)'),
         (args['num_zarcs']>=1, '{} W_c_zarc_1 (rad/s)'),
         (args['num_zarcs']>=2, '{} W_c_zarc_2 (rad/s)'),
         (args['num_zarcs']>=3, '{} W_c_zarc_3 (rad/s)'),
         (True, '{} Phi_warburg (unitless)'),
         (args['num_zarcs']>=1, '{} Phi_zarc_1 (unitless)'),
         (args['num_zarcs']>=2, '{} Phi_zarc_2 (unitless)'),
         (args['num_zarcs']>=3, '{} Phi_zarc_3 (unitless)'),
         (args['inductance'], '{} Phi_inductance (unitless)'),
         (args['zarc_inductance'], '{} Phi_zarc_inductance (unitless)'),
    ]


    with open(os.path.join(args['output_dir'], 'CircuitParameterFits.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(
            ['Original Filename'] +
            sum([
                [
                    t[1].format(lab) for t in all_labels if t[0]
                ] for lab in ['Inverse Model','Finetuned','Delta']
            ]),
        )
        for res in results:
            writer.writerow(
                [res['filename']] +
                sum([
                        [
                            '{}'.format(res[k][i]) for i in range(len(all_labels)) if all_labels[i][0]
                        ] for k in ['inverse_model_params', 'finetuned_params', 'delta_params']
                    ]
                )
            )





class Command(BaseCommand):
    """

    This is where the commandline arguments are interpreted and the appropriate function is called.
    """
    def add_arguments(self, parser):
        parser.add_argument('--mode', choices=[
                                               'import_process_output'
                                               ])
        parser.add_argument('--logdir')
        parser.add_argument('--dataset',default='USER1')
        parser.add_argument('--input_dir', default='import_test')
        parser.add_argument('--output_dir', default='OutputData')
        parser.add_argument('--seed', type=int, default=13311772)
        parser.add_argument('--nll_coeff', type=float, default=.1)
        parser.add_argument('--ordering_coeff', type=float, default=.5)
        parser.add_argument('--simplicity_coeff', type=float, default=.1)
        parser.add_argument('--sensible_phi_coeff', type=float, default=1.)
        parser.add_argument('--total_steps', type=int, default=1001)
        parser.add_argument('--learning_rate', type=float, default=1e-1)

        parser.add_argument('--angular-freq', dest='angular_freq', action='store_true')
        parser.add_argument('--no-angular-freq', dest='angular_freq', action='store_false')
        parser.set_defaults(angular_freq=False)

        parser.add_argument('--chunk_num', type=int, default=256)

        parser.add_argument('--inductance', dest='inductance', action='store_true')
        parser.add_argument('--no-inductance', dest='inductance', action='store_false')
        parser.set_defaults(inductance=False)

        parser.add_argument('--zarc-inductance', dest='zarc_inductance', action='store_true')
        parser.add_argument('--no-zarc-inductance', dest='zarc_inductance', action='store_false')
        parser.set_defaults(zarc_inductance=False)

        parser.add_argument('--num_zarcs', type=int, default=3)

    def handle(self, *args, **options):
        if options['mode'] == 'import_directory':
            import_directory(options)
        if options['mode'] == 'run_inverse_model_on_user_spectra':
            run_inverse_model_on_user_spectra(options)
        if options['mode'] == 'finetune':
            finetune(options)
        if options['mode'] == 'import_process_output':
            import_process_output(options)
