from django.db import models

# Create your models here.
# This allows us to place spectra into different datasets and treat them differently.
class Dataset(models.Model):
    label = models.CharField(max_length=100)
    def display(self):
        return "(Dataset {}, label {})".format(
            self.id,
            self.label,

        )

class AutomaticActiveSample(models.Model):
    freqs_with_tails_im_z = models.IntegerField(default=0)
    freqs_with_negative_im_z = models.IntegerField(default=0)
    sample_count = models.IntegerField(default=0)

    def display(self):
        return "(AAS {}, freqs_with_tails_im_z {}, freqs_with_negative_im_z {}, sample_count {})".format(
            self.id,
            self.freqs_with_tails_im_z,
            self.freqs_with_negative_im_z,
            self.sample_count)
    @property
    def sample_to_delete_count(self):
        return max(self.freqs_with_tails_im_z,
                                           int(float(self.freqs_with_negative_im_z) / 2.))
    @property
    def sample_to_keep_count(self):
        return self.sample_count - self.sample_to_delete_count

#The base class for a spectrum
class EISSpectrum(models.Model):
    filename = models.CharField(max_length=1000)

    active = models.BooleanField(default=True)
    dataset = models.ForeignKey(Dataset,
                                on_delete=models.SET_NULL,
                                null=True)
    automatic_active_sample = models.OneToOneField(
        AutomaticActiveSample,
        on_delete=models.SET_NULL,
        null=True
    )
    def __str__(self):
        return "Spectrum {}".format(self.filename)
    def __unicode__(self):
        return u'Spectrum {}'.format(self.filename)
    def display(self):
        dataset_disp = 'None'
        if not self.dataset is None:
            dataset_disp = self.dataset.display()

        automatic_active_sample_disp = 'None'
        if not self.automatic_active_sample is None:
            automatic_active_sample_disp = self.automatic_active_sample.display()

        return "(Spectrum {}, filename {}, active {}, dataset {}, automatic_active_sample {})".format(
            self.id,
            self.filename,
            self.active,
            dataset_disp,
            automatic_active_sample_disp,)



#a tuple (w, re[z], im[z])
class ImpedanceSample(models.Model):
    spectrum = models.ForeignKey(EISSpectrum, on_delete=models.CASCADE)
    log_ang_freq = models.FloatField()
    real_part = models.FloatField()
    imag_part = models.FloatField()
    active = models.BooleanField(default=True)

    def display(self):
        spectrum_disp = 'None'
        if not self.spectrum is None:
            spectrum_disp = self.spectrum.display()

        return "(ImpedanceSample {}, spectrum {}, log_ang_freq {}, real_part {}, imag_part {}, active {})".format(
            self.id,
            spectrum_disp,
            self.log_ang_freq,
            self.real_part,
            self.imag_part,
            self.active

        )
