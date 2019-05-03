from django.core.management.base import BaseCommand
from EIS.models import *

models_dict = {
    'Dataset':Dataset,
    'AutomaticActiveSample':AutomaticActiveSample,
    'EISSpectrum':EISSpectrum,
    'ImpedanceSample':ImpedanceSample,
    'InverseModel':InverseModel,
    'ShiftScaleParameters':ShiftScaleParameters,
}

class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--mode', choices=['add_default_datasets',
                                               'add_default_inverse_models',
                                               'display',
                                               'clear_all'
                                               ])
        parser.add_argument('--model', choices=list(models_dict.keys()) + ['*',''],default='')

    def handle(self, *args, **options):

        if options['mode'] == 'display':
            for element in models_dict[options['model']].objects.all():
                print('{}: {}'.format(options['model'], element.display()))

        if options['mode'] == 'clear_all':
            if options['model'] == '*':
                for key in models_dict.keys():

                    for element in models_dict[key].objects.all():
                        element.delete()

            else:
                for element in models_dict[options['model']].objects.all():
                    element.delete()

        if options['mode'] == 'add_default_datasets':

            dummies = [
                {
                    'label': 'EIS',
                },
                {
                    'label': 'FRA',
                },
                {
                    'label': 'USER',
                },
            ]
            for dummy in dummies:
                if not Dataset.objects.filter(label=dummy['label']):
                    dataset = Dataset(label=dummy['label'])
                    dataset.save()
                else:
                    for dataset in Dataset.objects.filter(label=dummy['label']):
                        dataset.label = dummy['label']
                        dataset.save()



        if options['mode'] == 'add_default_inverse_models':

            dummies = [
                {
                    'logdir': 'OnePercentTraining',
                    'kernel_size':7,
                    'conv_filters':16,
                    'num_conv':2,
                },
            ]
            for dummy in dummies:
                if not InverseModel.objects.filter(logdir=dummy['logdir']):
                    inv_model = InverseModel(
                        logdir=dummy['logdir'],
                        kernel_size=dummy['kernel_size'],
                        conv_filters=dummy['conv_filters'],
                        num_conv=dummy['num_conv'],
                    )
                    inv_model.save()
                else:
                    for inv_model in InverseModel.objects.filter(logdir=dummy['logdir']):
                        inv_model.logdir=dummy['logdir'],
                        inv_model.kernel_size=dummy['kernel_size'],
                        inv_model.conv_filters=dummy['conv_filters'],
                        inv_model.num_conv=dummy['num_conv'],
                        inv_model.save()