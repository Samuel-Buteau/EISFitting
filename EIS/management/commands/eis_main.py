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
from EIS.models import EISSpectrum,ImpedanceSample, AutomaticActiveSample, Dataset

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


class Command(BaseCommand):
    """

    This is where the commandline arguments are interpreted and the appropriate function is called.
    """
    def add_arguments(self, parser):
        parser.add_argument('--mode', choices=['import_directory',
                                               ])
        parser.add_argument('--logdir')
        parser.add_argument('--input_dir', default='import_test')
        parser.add_argument('--output_dir', default='OutputData')

    def handle(self, *args, **options):
        if options['mode'] == 'import_directory':
            import_directory(options)
