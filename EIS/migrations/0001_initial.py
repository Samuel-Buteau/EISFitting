# Generated by Django 2.2.13 on 2020-07-20 21:47

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ActivitySetting',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('useless', models.BooleanField(default=True)),
            ],
        ),
        migrations.CreateModel(
            name='AutomaticActiveSample',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('freqs_with_tails_im_z', models.IntegerField(default=0)),
                ('freqs_with_negative_im_z', models.IntegerField(default=0)),
                ('sample_count', models.IntegerField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name='CircuitParameterSet',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('circuit', models.CharField(max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('label', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='EISSpectrum',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('filename', models.CharField(max_length=1000)),
                ('active', models.BooleanField(default=True)),
                ('automatic_active_sample', models.OneToOneField(null=True, on_delete=django.db.models.deletion.SET_NULL, to='EIS.AutomaticActiveSample')),
                ('dataset', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='EIS.Dataset')),
            ],
        ),
        migrations.CreateModel(
            name='FileFormat',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('extension', models.CharField(max_length=10)),
            ],
        ),
        migrations.CreateModel(
            name='FitSpectrum',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('active', models.BooleanField(default=True)),
            ],
        ),
        migrations.CreateModel(
            name='InverseModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('logdir', models.CharField(max_length=100)),
                ('kernel_size', models.IntegerField(default=7)),
                ('conv_filters', models.IntegerField(default=16)),
                ('num_conv', models.IntegerField(default=2)),
            ],
        ),
        migrations.CreateModel(
            name='ShiftScaleParameters',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('r_alpha', models.FloatField(default=0)),
                ('w_alpha', models.FloatField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name='InverseModelResult',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('inductance', models.BooleanField(default=False)),
                ('zarc_inductance', models.BooleanField(default=False)),
                ('warburg_inception', models.BooleanField(default=False)),
                ('num_zarcs', models.IntegerField(default=3)),
                ('activity_setting', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='EIS.ActivitySetting')),
                ('circuit_parameters', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='EIS.CircuitParameterSet')),
                ('fit_spectrum', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='EIS.FitSpectrum')),
                ('inv_model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='EIS.InverseModel')),
                ('shift_scale_parameters', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='EIS.ShiftScaleParameters')),
                ('spectrum', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='EIS.EISSpectrum')),
            ],
        ),
        migrations.CreateModel(
            name='ImpedanceSample',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('log_ang_freq', models.FloatField()),
                ('real_part', models.FloatField()),
                ('imag_part', models.FloatField()),
                ('active', models.BooleanField(default=True)),
                ('spectrum', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='EIS.EISSpectrum')),
            ],
        ),
        migrations.CreateModel(
            name='FitSample',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('log_ang_freq', models.FloatField()),
                ('real_part', models.FloatField()),
                ('imag_part', models.FloatField()),
                ('fit', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='EIS.FitSpectrum')),
            ],
        ),
        migrations.CreateModel(
            name='FinetuneResult',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('learning_rate', models.FloatField(default=0.001)),
                ('nll_coeff', models.FloatField(default=0.1)),
                ('ordering_coeff', models.FloatField(default=0.5)),
                ('simplicity_coeff', models.FloatField(default=0.1)),
                ('sensible_phi_coeff', models.FloatField(default=1.0)),
                ('circuit_parameters', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='EIS.CircuitParameterSet')),
                ('fit_spectrum', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='EIS.FitSpectrum')),
                ('inv_model_result', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='EIS.InverseModelResult')),
            ],
        ),
        migrations.AddField(
            model_name='eisspectrum',
            name='file_format',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='EIS.FileFormat'),
        ),
        migrations.CreateModel(
            name='CircuitParameter',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('index', models.IntegerField(default=0)),
                ('value', models.FloatField(default=0)),
                ('set', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='EIS.CircuitParameterSet')),
            ],
        ),
        migrations.CreateModel(
            name='ActivitySample',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('active', models.BooleanField(default=True)),
                ('sample', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='EIS.ImpedanceSample')),
                ('setting', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='EIS.ActivitySetting')),
            ],
        ),
    ]
