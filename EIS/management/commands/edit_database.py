from django.core.management.base import BaseCommand
from EIS.models import *

models_dict = {
    'Dataset':Dataset,
    'AutomaticActiveSample':AutomaticActiveSample,
    'EISSpectrum':EISSpectrum,
    'ImpedanceSample':ImpedanceSample,
}

class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--mode', choices=['add_default_datasets',
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
