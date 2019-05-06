import contextlib
import csv
import math
import os
import pickle
import random

import copy
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from django.core.management.base import BaseCommand
import re

from import_eis_files import import_eis_file
from EIS.models import *

from matplotlib.gridspec import GridSpec


def import_directory(args):

    '''
    For now all the files imported go to the USER dataset, and the results come from the USER dataset.

    :param args:
    :return:
    '''

    if not os.path.exists(args['input_dir']):
        print('Please provide a valid value for --input_dir')
        return

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
            '''
            TODO: figure out how to create in bulk. I think this requires me to query the primary keys.
            '''
            aas = AutomaticActiveSample(
                freqs_with_tails_im_z=record['freqs_with_tails_im_z'],
                freqs_with_negative_im_z=record['freqs_with_negative_im_z'],
                sample_count = len(record['original_spectrum'][0])
            )
            aas.save()

            dataset = Dataset.objects.get(label='USER')

            new_spectrum = EISSpectrum(filename=str(filename),
                                       dataset = dataset,
                                       automatic_active_sample = aas,
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


def get_inv_model_results_or_none(spectrum, inv_model):
    '''

    returns two values. first is the inv_model_results. second is a bool saying if it existed.
    :param spectrum:
    :param inv_model:
    :return:

    '''
    if InverseModelResult.objects.filter(spectrum=spectrum, inv_model=inv_model).exists():
        for inv_model_res in InverseModelResult.objects.filter(spectrum=spectrum, inv_model=inv_model):
            match = True
            for activity_sample in ActivitySample.objects.filter(setting=inv_model_res.activity_setting):
                if not activity_sample.active == activity_sample.sample.active:
                    match = False
                    break

            if match == True:
                return inv_model_res, True

    return None, False



from EISFittingModelDefinitions import Prior,ParameterVAE, shift_scale_param_extract,normalized_spectrum,original_spectrum, initialize_session,restore_params,deparameterized_params




def run_inverse_model_on_user_spectra(args):
    user_dataset = Dataset.objects.get(label='USER')

    if not InverseModel.objects.filter(logdir=args['logdir']).exists():
        raise Exception('Parameter --logdir does not correspond to a valid model.')
    inv_model = InverseModel.objects.get(logdir=args['logdir'])
    print('Now using {}.'.format(inv_model.display()))



    random.seed(a=args['seed'])
    batch_size = tf.placeholder(dtype=tf.int32)
    prior_mu, prior_log_sigma_sq = Prior()

    frequencies = tf.placeholder(shape=[None, None], dtype=tf.float32)
    input_impedances = tf.placeholder(shape=[None, None,2], dtype=tf.float32)

    inputs = tf.concat([tf.expand_dims(frequencies, axis=2), input_impedances], axis=2)

    number_of_zarcs = 3
    number_of_params = 1 + 1 + number_of_zarcs + 1 + 1 + 1 + number_of_zarcs + 1 + number_of_zarcs + 1 + 1

    model = ParameterVAE(kernel_size=inv_model.kernel_size, conv_filters=inv_model.conv_filters,
                          num_conv=inv_model.num_conv, trainable=False, num_encoded=number_of_params)

    impedances, representation_mu = \
        model.build_forward(inputs=inputs, batch_size=batch_size, priors=prior_mu)




    cleaned_data = []


    activity_sample_list = []
    for spectrum in user_dataset.eisspectrum_set.all():
        _, already_exists = get_inv_model_results_or_none(spectrum,inv_model)
        if already_exists:
            continue

        list_of_samples = spectrum.impedancesample_set.filter(active=True).order_by('log_ang_freq')




        spec_numpy = numpy.array([[samp.log_ang_freq, samp.real_part, samp.imag_part] for samp in list_of_samples if samp.active])

        spec_tuple = (spec_numpy[:,0],spec_numpy[:,1],spec_numpy[:,2])
        shift_scale_params = shift_scale_param_extract(spec_tuple)



        ssp = ShiftScaleParameters(
            r_alpha=shift_scale_params['r_alpha'],
            w_alpha=shift_scale_params['w_alpha'],
        )
        ssp.save()

        cp = CircuitParameterSet(
            circuit = 'standard3zarc'
        )
        cp.save()

        fs = FitSpectrum()
        fs.save()

        activity_setting = ActivitySetting()
        activity_setting.save()

        inverse_model_results = InverseModelResult(
            spectrum = spectrum,
            inv_model = inv_model,
            activity_setting=activity_setting,
            shift_scale_parameters = ssp,
            circuit_parameters = cp,
            fit_spectrum = fs,
        )
        inverse_model_results.save()

        for samp in list_of_samples:
            activity_sample_list.append(
                ActivitySample(
                    setting=activity_setting,
                    sample = samp,
                    active = samp.active,
                )
            )


        log_freq, re_z, im_z = normalized_spectrum(spec_tuple, params=inverse_model_results.shift_scale_parameters.to_dict())

        cleaned_data.append((log_freq,re_z,im_z,inverse_model_results.id))


    ActivitySample.objects.bulk_create(activity_sample_list)

    cleaned_data = sorted(cleaned_data, key=lambda x: len(x[0]))

    grouped_data = []

    current_group =[]
    for freq, re_z, im_z, id in cleaned_data:
        current_len = len(freq)
        if len(current_group) == 0:
            current_group.append((freq, re_z, im_z, id ))
        elif current_len == len(current_group[0][0]):
            current_group.append((freq, re_z,im_z, id ))
            if len(current_group) == 64:
                grouped_data.append(copy.deepcopy(current_group))
                current_group = []
        else:
            grouped_data.append(copy.deepcopy(current_group))
            current_group = [(freq, re_z, im_z, id )]

    if not len(current_group) == 0:
        grouped_data.append(current_group)


    # this is where we run the inverse model and then record the results.
    results = []
    with initialize_session(logdir=inv_model.logdir, seed=args['seed']) as (sess, saver):
        for g in grouped_data:
            batch_len = len(g)
            batch_frequecies = numpy.array([x[0] for x in g])
            batch_impedances =  numpy.array([numpy.stack((x[1], x[2]), axis=1) for x in g])
            batch_ids = numpy.array([x[3] for x in g])

            out_impedance,in_impedance, freqs, representation_mu_value  = \
                sess.run([impedances, input_impedances, frequencies, representation_mu],
                         feed_dict={batch_size: batch_len,
                                    model.dropout: 0.0,
                                    frequencies: batch_frequecies,
                                    input_impedances: batch_impedances
                                    })


            current_results = [
                (freqs[index],
                 in_impedance[index],
                 out_impedance[index],
                 representation_mu_value[index],
                 batch_ids[index])
                for index in range(batch_len)
            ]
            results += copy.deepcopy(current_results)

        circuit_parameters_list = []
        fit_samples_list = []
        for freqs, in_impedance,out_impedance, representation_mu_value, id in results:
            inv_model_result = InverseModelResult.objects.get(id=id)

            for i in range(len(representation_mu_value)):
                circuit_parameters_list.append(
                    CircuitParameter(
                        set=inv_model_result.circuit_parameters,
                        index = i,
                        value = representation_mu_value[i]
                    )
                )

            for i in range(len(freqs)):
                fit_samples_list.append(
                    FitSample(
                        fit=inv_model_result.fit_spectrum,
                        log_ang_freq = freqs[i],
                        real_part = out_impedance[i,0],
                        imag_part = out_impedance[i,1],
                    )
                )
        CircuitParameter.objects.bulk_create(circuit_parameters_list)
        FitSample.objects.bulk_create(fit_samples_list)


    # This is where we do a test display of the results.
    for spectrum in user_dataset.eisspectrum_set.all():
        inv_model_result, should_be_true = get_inv_model_results_or_none(spectrum,inv_model)
        if not should_be_true:
            raise Exception('should be true but is false. spectrum was {}, inv_model was {}'.format(spectrum,inv_model))

        orig_spec = numpy.array([[samp.log_ang_freq, samp.real_part, samp.imag_part] for samp in spectrum.impedancesample_set.order_by('log_ang_freq')])
        orig_spec[:,0] = 1. / (2. * numpy.pi) * numpy.exp(orig_spec[:,0])

        fit_spec_unnorm = numpy.array([[samp.log_ang_freq, samp.real_part, samp.imag_part] for samp in inv_model_result.fit_spectrum.fitsample_set.order_by('log_ang_freq')])
        freq_fit, real_fit, imag_fit = original_spectrum((fit_spec_unnorm[:,0],fit_spec_unnorm[:,1],fit_spec_unnorm[:,2], ), inv_model_result.shift_scale_parameters.to_dict())
        fit_spec = numpy.stack((freq_fit, real_fit, imag_fit), axis=-1)
        fit_spec[:, 0] = 1. / (2. * numpy.pi) * numpy.exp(fit_spec[:, 0])



        fig = plt.figure()
        gs = GridSpec(2, 2, figure=fig)

        ax = fig.add_subplot(gs[0,:])
        ax.scatter(orig_spec[:,1],-orig_spec[:,2], c='k')
        ax.plot(fit_spec[:,1],-fit_spec[:,2], c='r')

        ax = fig.add_subplot(gs[1,0])
        ax.set_xscale('log')
        ax.scatter(orig_spec[:, 0], -orig_spec[:, 2], c='k')
        ax.plot(fit_spec[:, 0], -fit_spec[:, 2], c='r')

        ax = fig.add_subplot(gs[1,1])
        ax.set_xscale('log')
        ax.scatter(orig_spec[:, 0], orig_spec[:, 1], c='k')
        ax.plot(fit_spec[:, 0], fit_spec[:, 1], c='r')

        fig.tight_layout(h_pad=0., w_pad=0.)
        plt.show()

        print(inv_model_result.get_circuit_parameters_in_original_form())






class Command(BaseCommand):
    """

    This is where the commandline arguments are interpreted and the appropriate function is called.
    """
    def add_arguments(self, parser):
        parser.add_argument('--mode', choices=['import_directory',
                                               'run_inverse_model_on_user_spectra'
                                               ])
        parser.add_argument('--logdir')
        parser.add_argument('--input_dir', default='import_test')
        parser.add_argument('--output_dir', default='OutputData')
        parser.add_argument('--seed', type=int, default=13311772)

    def handle(self, *args, **options):
        if options['mode'] == 'import_directory':
            import_directory(options)
        if options['mode'] == 'run_inverse_model_on_user_spectra':
            run_inverse_model_on_user_spectra(options)
