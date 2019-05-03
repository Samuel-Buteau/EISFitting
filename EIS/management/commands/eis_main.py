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



from EISFittingModelDefinitions import Prior,ParameterVAE, shift_scale_param_extract,normalized_spectrum,initialize_session
def run_inverse_model_on_user_spectra(args):
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

    for spectrum in EISSpectrum.objects.filter(dataset__isnull=False).filter(dataset__label='USER'):

        list_of_samples = ImpedanceSample.objects.filter(spectrum=spectrum, active=True).order_by('log_ang_freq')

        spec_numpy = numpy.array([[samp.log_ang_freq, samp.real_part, samp.imag_part] for samp in list_of_samples if samp.active])

        spec_tuple = (spec_numpy[:,0],spec_numpy[:,1],spec_numpy[:,2])
        shift_scale_params = shift_scale_param_extract(spec_tuple)
        ssp = ShiftScaleParameters(
            r_alpha=shift_scale_params['r_alpha'],
            w_alpha=shift_scale_params['w_alpha'],
        )
        ssp.save()
        spectrum.shift_scale_parameters = ssp
        spectrum.save()



        log_freq, re_z, im_z = normalized_spectrum(spec_tuple, params=spectrum.shift_scale_parameters.to_dict())

        cleaned_data.append((log_freq,re_z,im_z,spectrum.id))

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



    results = []
    with initialize_session(logdir=inv_model.logdir, seed=args['seed']) as (sess, saver):
        for g in grouped_data:
            batch_len = len(g)
            batch_frequecies = numpy.array([x[0] for x in g])
            batch_impedances =  numpy.array([numpy.stack((x[1], x[2]), axis=1) for x in g])
            batch_ids = numpy.array([x[3] for x in g])

            out_impedance,in_impedance, freqs, representation_mu_value  = \
                sess.run([  impedances, input_impedances, frequencies, representation_mu],
                         feed_dict={batch_size: batch_len,
                                    model.dropout: 0.0,
                                    frequencies: batch_frequecies,
                                    input_impedances: batch_impedances
                                    })


            current_results = [(freqs[index], in_impedance[index],out_impedance[index], representation_mu_value[index], batch_ids[index]) for index in range(batch_len)]
            results += copy.deepcopy(current_results)


        with open(os.path.join(".", args['output_dir'], 'inverse_model_on_user_spectra_results.file'), 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)




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
