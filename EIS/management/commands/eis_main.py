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




from EISFittingModelDefinitions import Prior,ParameterVAE, shift_scale_param_extract,normalized_spectrum,original_spectrum, initialize_session,NonparametricOptimizer






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






def finetune(args):
    number_of_zarcs = 3
    number_of_params = 1 + 1 + number_of_zarcs + 1 + 1 + 1 + number_of_zarcs + 1 + number_of_zarcs + 1 + 1


    user_dataset = Dataset.objects.get(label='USER')

    if not InverseModel.objects.filter(logdir=args['logdir']).exists():
        raise Exception('Parameter --logdir does not correspond to a valid model.')
    inv_model = InverseModel.objects.get(logdir=args['logdir'])
    print('Now using {}.'.format(inv_model.display()))
    max_len = 0
    spectrum_count = 0
    for spectrum in user_dataset.eisspectrum_set.all():
        inv_model_result, already_exists = get_inv_model_results_or_none(spectrum, inv_model)
        assert already_exists

        _, already_exists = get_finetune_results_or_none(inv_model_result,args)
        if not already_exists:
            spectrum_count += 1
            max_len = max(max_len,spectrum.impedancesample_set.filter(active=True).count())

    # TODO: make sure that you run on all the inverse_model_results that dont have a finetune result.


    #must be kept in sync
    all_spectra = numpy.zeros(shape=(spectrum_count, max_len, 3), dtype=numpy.float32)
    all_ids = numpy.zeros(shape=(spectrum_count), dtype=numpy.int32)
    all_masks = numpy.zeros(shape=(spectrum_count, max_len), dtype=numpy.float32)
    all_params = numpy.zeros(shape=(spectrum_count, number_of_params), dtype=numpy.float32)
    all_extrema_freqs = numpy.zeros(shape=(spectrum_count,2), dtype=numpy.float32)

    main_index = 0
    for spectrum in user_dataset.eisspectrum_set.all():

        inv_model_result, already_exists = get_inv_model_results_or_none(spectrum, inv_model)
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
            n_arr = len(norm_arr)

            all_spectra[main_index,:n_arr, :] = norm_arr
            all_ids[main_index] = finetune_results.id
            all_masks[main_index, :n_arr] = numpy.ones(shape=(n_arr), dtype=numpy.float32)
            all_params[main_index,:] = inv_model_result.circuit_parameters.get_parameter_array()
            all_extrema_freqs[main_index, 0] = numpy.min(norm_arr[:,0])
            all_extrema_freqs[main_index, 1] = numpy.max(norm_arr[:, 0])

            main_index += 1

    print(all_spectra)
    print(all_ids)
    print(all_masks)
    print(all_params)
    print(all_extrema_freqs)

    # build the computation graph
    batch_size = tf.placeholder(dtype=tf.int32)
    prior_mu, prior_log_sigma_sq = Prior()


    indices = tf.placeholder(shape=[None], dtype=tf.int32)




    model = NonparametricOptimizer(
        parameter_matrix=all_params,
        spectrum_matrix=all_spectra,
        mask_matrix=all_masks,
        extrema_freqs_matrix = all_extrema_freqs,
    )



    loss, train_step, impedances, representation_mu, my_reconstruction_loss = \
        model.optimize_direct(
            indices=indices ,prior_mu=prior_mu,
            prior_log_sigma_sq=prior_log_sigma_sq,
            learning_rate=args['learning_rate'],
            batch_size=batch_size
        )

    indexed_matrices = \
        model.get_indexed_matrices(
            indices=indices,
            batch_size=batch_size
        )

    step = tf.train.get_or_create_global_step()
    increment_step = step.assign_add(1)

    # for now, batch_size is just the total size

    actual_batch_size = spectrum_count


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        while True:
            current_step = sess.run(step)
            if current_step >= args['total_steps']:
                print('Training complete.')
                break

            loss_value, _, step_value = \
                sess.run([  loss, train_step, increment_step],
                         feed_dict={
                             batch_size: actual_batch_size,
                             indices: range(actual_batch_size),
                             model.sensible_phi_coeff: args['sensible_phi_coeff'],
                             model.simplicity_coeff: args['simplicity_coeff'],
                             model.nll_coeff: args['nll_coeff'],
                             model.ordering_coeff: args['ordering_coeff']
                                    })


            if (current_step % 100) ==0:
                print('iteration {}, total loss {}'.format(current_step, loss_value))
            if current_step == 1000:
                freq_val,in_val,out_val, mask_val, param_val = \
                    sess.run([
                        indexed_matrices['frequencies'],
                        indexed_matrices['in_impedances'],
                        indexed_matrices['out_impedances'],
                        indexed_matrices['masks'],
                        indexed_matrices['parameters'],
                    ],
                             feed_dict={
                                 batch_size: actual_batch_size,
                                 indices: range(actual_batch_size),
                             })

                ids_val = [all_ids[ind] for ind in range(actual_batch_size)]

                circuit_parameters_list = []
                fit_samples_list = []

                for ind in range(actual_batch_size):


                    finetune_results = FinetuneResult.objects.get(id=ids_val[ind])

                    for i in range(len(param_val[ind])):
                        circuit_parameters_list.append(
                            CircuitParameter(
                                set=finetune_results.circuit_parameters,
                                index=i,
                                value=param_val[ind,i]
                            )
                        )

                    for i in range(len(freq_val[ind])):
                        if int(mask_val[ind,i]) == 0:
                            break

                        fit_samples_list.append(
                            FitSample(
                                fit=finetune_results.fit_spectrum,
                                log_ang_freq=freq_val[ind, i],
                                real_part=out_val[ind, i, 0],
                                imag_part=out_val[ind, i, 1],
                            )
                        )
                CircuitParameter.objects.bulk_create(circuit_parameters_list)
                FitSample.objects.bulk_create(fit_samples_list)

            #TODO: visualize the results to see if something is wrong.


    # This is where we do a test display of the results.
    for spectrum in user_dataset.eisspectrum_set.all():
        inv_model_result, should_be_true = get_inv_model_results_or_none(spectrum,inv_model)
        if not should_be_true:
            raise Exception('should be true but is false. spectrum was {}, inv_model was {}'.format(spectrum,inv_model))

        finetune_results, already_exists = get_finetune_results_or_none(inv_model_result, args)
        assert already_exists


        orig_spec = numpy.array([[samp.log_ang_freq, samp.real_part, samp.imag_part] for samp in spectrum.impedancesample_set.order_by('log_ang_freq')])
        orig_spec[:,0] = 1. / (2. * numpy.pi) * numpy.exp(orig_spec[:,0])

        fit_spec_unnorm = numpy.array([[samp.log_ang_freq, samp.real_part, samp.imag_part] for samp in inv_model_result.fit_spectrum.fitsample_set.order_by('log_ang_freq')])
        freq_fit, real_fit, imag_fit = original_spectrum((fit_spec_unnorm[:,0],fit_spec_unnorm[:,1],fit_spec_unnorm[:,2], ), inv_model_result.shift_scale_parameters.to_dict())
        fit_spec = numpy.stack((freq_fit, real_fit, imag_fit), axis=-1)
        fit_spec[:, 0] = 1. / (2. * numpy.pi) * numpy.exp(fit_spec[:, 0])

        fit_spec_unnorm2 = numpy.array([[samp.log_ang_freq, samp.real_part, samp.imag_part] for samp in
                                       finetune_results.fit_spectrum.fitsample_set.order_by('log_ang_freq')])
        freq_fit2, real_fit2, imag_fit2 = original_spectrum(
            (fit_spec_unnorm2[:, 0], fit_spec_unnorm2[:, 1], fit_spec_unnorm2[:, 2],),
            inv_model_result.shift_scale_parameters.to_dict())
        fit_spec2 = numpy.stack((freq_fit2, real_fit2, imag_fit2), axis=-1)
        fit_spec2[:, 0] = 1. / (2. * numpy.pi) * numpy.exp(fit_spec2[:, 0])



        fig = plt.figure()
        gs = GridSpec(2, 2, figure=fig)

        ax = fig.add_subplot(gs[0,:])
        ax.scatter(orig_spec[:,1],-orig_spec[:,2], c='k')
        ax.plot(fit_spec[:,1],-fit_spec[:,2], c='r')
        ax.plot(fit_spec2[:, 1], -fit_spec2[:, 2], c='b')

        ax = fig.add_subplot(gs[1,0])
        ax.set_xscale('log')
        ax.scatter(orig_spec[:, 0], -orig_spec[:, 2], c='k')
        ax.plot(fit_spec[:, 0], -fit_spec[:, 2], c='r')
        ax.plot(fit_spec2[:, 0], -fit_spec2[:, 2], c='b')

        ax = fig.add_subplot(gs[1,1])
        ax.set_xscale('log')
        ax.scatter(orig_spec[:, 0], orig_spec[:, 1], c='k')
        ax.plot(fit_spec[:, 0], fit_spec[:, 1], c='r')
        ax.plot(fit_spec2[:, 0], fit_spec2[:, 1], c='b')

        fig.tight_layout(h_pad=0., w_pad=0.)
        plt.show()

        print(inv_model_result.get_circuit_parameters_in_original_form())
        print(finetune_results.get_circuit_parameters_in_original_form())



def import_process_output(args):
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

    random.seed(a=args['seed'])
    batch_size = tf.placeholder(dtype=tf.int32)
    prior_mu, prior_log_sigma_sq = Prior()

    frequencies = tf.placeholder(shape=[None, None], dtype=tf.float32)
    input_impedances = tf.placeholder(shape=[None, None, 2], dtype=tf.float32)

    inputs = tf.concat([tf.expand_dims(frequencies, axis=2), input_impedances], axis=2)

    number_of_zarcs = 3
    number_of_params = 1 + 1 + number_of_zarcs + 1 + 1 + 1 + number_of_zarcs + 1 + number_of_zarcs + 1 + 1

    model = ParameterVAE(kernel_size=inv_model.kernel_size, conv_filters=inv_model.conv_filters,
                         num_conv=inv_model.num_conv, trainable=False, num_encoded=number_of_params)

    impedances, representation_mu = \
        model.build_forward(inputs=inputs, batch_size=batch_size, priors=prior_mu)

    cleaned_data = []

    activity_sample_list = []
    for spectrum in user_dataset.eisspectrum_set.filter(active=True):
        if not spectrum.any_active():
            continue

        _, already_exists = get_inv_model_results_or_none(spectrum, inv_model)
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

    cleaned_data = sorted(cleaned_data, key=lambda x: len(x[0]))

    grouped_data = []

    current_group = []
    for freq, re_z, im_z, id in cleaned_data:
        current_len = len(freq)
        if len(current_group) == 0:
            current_group.append((freq, re_z, im_z, id))
        elif current_len == len(current_group[0][0]):
            current_group.append((freq, re_z, im_z, id))
            if len(current_group) == 64:
                grouped_data.append(copy.deepcopy(current_group))
                current_group = []
        else:
            grouped_data.append(copy.deepcopy(current_group))
            current_group = [(freq, re_z, im_z, id)]

    if not len(current_group) == 0:
        grouped_data.append(current_group)

    # this is where we run the inverse model and then record the results.
    results = []
    with initialize_session(logdir=inv_model.logdir, seed=args['seed']) as (sess, saver):
        for g in grouped_data:
            batch_len = len(g)
            batch_frequecies = numpy.array([x[0] for x in g])
            batch_impedances = numpy.array([numpy.stack((x[1], x[2]), axis=1) for x in g])
            batch_ids = numpy.array([x[3] for x in g])

            out_impedance, in_impedance, freqs, representation_mu_value = \
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


    max_len = 0
    spectrum_count = 0
    for spectrum in user_dataset.eisspectrum_set.filter(active=True):
        if not spectrum.any_active():
            continue

        inv_model_result, already_exists = get_inv_model_results_or_none(spectrum, inv_model)
        assert already_exists

        _, already_exists = get_finetune_results_or_none(inv_model_result, args)
        if not already_exists:
            spectrum_count += 1
            max_len = max(max_len, spectrum.impedancesample_set.filter(active=True).count())

    if not spectrum_count == 0:

        # must be kept in sync
        all_spectra = numpy.zeros(shape=(spectrum_count, max_len, 3), dtype=numpy.float32)
        all_ids = numpy.zeros(shape=(spectrum_count), dtype=numpy.int32)
        all_valid_freqs_counts = numpy.zeros(shape=(spectrum_count), dtype=numpy.int32)
        all_params = numpy.zeros(shape=(spectrum_count, number_of_params), dtype=numpy.float32)

        main_index = 0
        for spectrum in user_dataset.eisspectrum_set.filter(active=True):
            if not spectrum.any_active():
                continue

            inv_model_result, already_exists = get_inv_model_results_or_none(spectrum, inv_model)
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
                n_arr = len(norm_arr)

                all_spectra[main_index, :n_arr, :] = norm_arr
                all_ids[main_index] = finetune_results.id
                all_valid_freqs_counts[main_index] = n_arr
                all_params[main_index, :] = inv_model_result.circuit_parameters.get_parameter_array()

                main_index += 1

        print(all_spectra)
        print(all_ids)
        print(all_params)


        # build the computation graph
        batch_size = tf.placeholder(dtype=tf.int32)
        prior_mu, prior_log_sigma_sq = Prior()

        indices = tf.placeholder(shape=[None], dtype=tf.int32)

        model = NonparametricOptimizer(
            parameter_matrix=all_params,
            spectrum_matrix=all_spectra,
            valid_freqs_counts_matrix=all_valid_freqs_counts,
        )

        loss, train_step, impedances, representation_mu, my_reconstruction_loss = \
            model.optimize_direct(
                indices=indices, prior_mu=prior_mu,
                prior_log_sigma_sq=prior_log_sigma_sq,
                learning_rate=args['learning_rate'],
                batch_size=batch_size
            )

        indexed_matrices = \
            model.get_indexed_matrices(
                indices=indices,
                batch_size=batch_size
            )

        step = tf.train.get_or_create_global_step()
        increment_step = step.assign_add(1)

        # for now, batch_size is just the total size

        actual_batch_size = spectrum_count
        full_index_list = range(actual_batch_size)
        if actual_batch_size < 2048:
            full_index_lists = numpy.array([full_index_list])
        else:
            num_chunks = 1 + int(actual_batch_size/2048)
            full_index_lists = numpy.array_split(full_index_list, num_chunks)

        for full_index_list in full_index_lists:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                while True:
                    current_step = sess.run(step)
                    if current_step >= args['total_steps']:
                        print('Training complete.')
                        break

                    loss_value, _, step_value = \
                        sess.run([loss, train_step, increment_step],
                                 feed_dict={
                                     batch_size: len(full_index_list),
                                     indices: full_index_list,
                                     model.sensible_phi_coeff: args['sensible_phi_coeff'],
                                     model.simplicity_coeff: args['simplicity_coeff'],
                                     model.nll_coeff: args['nll_coeff'],
                                     model.ordering_coeff: args['ordering_coeff']
                                 })

                    if (current_step % 100) == 0:
                        print('iteration {}, total loss {}'.format(current_step, loss_value))
                    if current_step == 1000:
                        freq_val, in_val, out_val, valid_freqs_counts_val, param_val = \
                            sess.run([
                                indexed_matrices['frequencies'],
                                indexed_matrices['in_impedances'],
                                indexed_matrices['out_impedances'],
                                indexed_matrices['valid_freqs_counts'],
                                indexed_matrices['parameters'],
                            ],
                                feed_dict={
                                    batch_size: len(full_index_list),
                                    indices: full_index_list,
                                })

                        ids_val = [all_ids[ind] for ind in full_index_list]

                        circuit_parameters_list = []
                        fit_samples_list = []

                        for ind in range(len(full_index_list)):

                            finetune_results = FinetuneResult.objects.get(id=ids_val[ind])

                            for i in range(len(param_val[ind])):
                                circuit_parameters_list.append(
                                    CircuitParameter(
                                        set=finetune_results.circuit_parameters,
                                        index=i,
                                        value=param_val[ind, i]
                                    )
                                )

                            for i in range(valid_freqs_counts_val[ind]):
                                fit_samples_list.append(
                                    FitSample(
                                        fit=finetune_results.fit_spectrum,
                                        log_ang_freq=freq_val[ind, i],
                                        real_part=out_val[ind, i, 0],
                                        imag_part=out_val[ind, i, 1],
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

        inv_model_result, should_be_true = get_inv_model_results_or_none(spectrum, inv_model)
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
    with open(os.path.join(args['output_dir'], 'CircuitParameterFits.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Original Filename',
                         'Inverse Model R (ohm)',
                         'Inverse Model R_zarc_impedance (ohm)',
                         'Inverse Model R_zarc_1 (ohm)',
                         'Inverse Model R_zarc_2 (ohm)',
                         'Inverse Model R_zarc_3 (ohm)',
                         'Inverse Model Q_warburg (?)',
                         'Inverse Model Q_inductance (?)',
                         'Inverse Model W_c_inductance (rad/s)',
                         'Inverse Model W_c_zarc_1 (rad/s)',
                         'Inverse Model W_c_zarc_2 (rad/s)',
                         'Inverse Model W_c_zarc_3 (rad/s)',
                         'Inverse Model Phi_warburg (unitless)',
                         'Inverse Model Phi_zarc_1 (unitless)',
                         'Inverse Model Phi_zarc_2 (unitless)',
                         'Inverse Model Phi_zarc_3 (unitless)',
                         'Inverse Model Phi_inductance (unitless)',
                         'Inverse Model Phi_zarc_inductance (unitless)',
                         'Finetuned R (ohm)',
                         'Finetuned R_zarc_impedance (ohm)',
                         'Finetuned R_zarc_1 (ohm)',
                         'Finetuned R_zarc_2 (ohm)',
                         'Finetuned R_zarc_3 (ohm)',
                         'Finetuned Q_warburg (?)',
                         'Finetuned Q_inductance (?)',
                         'Finetuned W_c_inductance (rad/s)',
                         'Finetuned W_c_zarc_1 (rad/s)',
                         'Finetuned W_c_zarc_2 (rad/s)',
                         'Finetuned W_c_zarc_3 (rad/s)',
                         'Finetuned Phi_warburg (unitless)',
                         'Finetuned Phi_zarc_1 (unitless)',
                         'Finetuned Phi_zarc_2 (unitless)',
                         'Finetuned Phi_zarc_3 (unitless)',
                         'Finetuned Phi_inductance (unitless)',
                         'Finetuned Phi_zarc_inductance (unitless)',
                         'Delta R (ohm)',
                         'Delta R_zarc_impedance (ohm)',
                         'Delta R_zarc_1 (ohm)',
                         'Delta R_zarc_2 (ohm)',
                         'Delta R_zarc_3 (ohm)',
                         'Delta Q_warburg (?)',
                         'Delta Q_inductance (?)',
                         'Delta W_c_inductance (rad/s)',
                         'Delta W_c_zarc_1 (rad/s)',
                         'Delta W_c_zarc_2 (rad/s)',
                         'Delta W_c_zarc_3 (rad/s)',
                         'Delta Phi_warburg (unitless)',
                         'Delta Phi_zarc_1 (unitless)',
                         'Delta Phi_zarc_2 (unitless)',
                         'Delta Phi_zarc_3 (unitless)',
                         'Delta Phi_inductance (unitless)',
                         'Delta Phi_zarc_inductance (unitless)',
                         ])
        for res in results:
            writer.writerow(
                [res['filename']] +
                ['{}'.format(x) for x in res['inverse_model_params']] +
                ['{}'.format(x) for x in res['finetuned_params']] +
                ['{}'.format(x) for x in res['delta_params']]
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

    def handle(self, *args, **options):
        if options['mode'] == 'import_directory':
            import_directory(options)
        if options['mode'] == 'run_inverse_model_on_user_spectra':
            run_inverse_model_on_user_spectra(options)
        if options['mode'] == 'finetune':
            finetune(options)
        if options['mode'] == 'import_process_output':
            import_process_output(options)
