# Generated by Django 2.1.9 on 2019-06-30 18:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('EIS', '0009_auto_20190508_1020'),
    ]

    operations = [
        migrations.AddField(
            model_name='inversemodelresult',
            name='inductance',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='inversemodelresult',
            name='num_zarcs',
            field=models.IntegerField(default=3),
        ),
        migrations.AddField(
            model_name='inversemodelresult',
            name='zarc_inductance',
            field=models.BooleanField(default=False),
        ),
    ]