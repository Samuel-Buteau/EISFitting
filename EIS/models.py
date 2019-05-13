from django.db import models
import numpy
from EISFittingModelDefinitions import restore_params,deparameterized_params,normalized_spectrum


'''
TODO:
    - create a inverse_model results class
    - move shift parameters to it
    - link it to the inverse model used.
    - link it to the spectrum used.
    - include the parameter set. 


'''
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

class CircuitParameterSet(models.Model):
    circuit = models.CharField(max_length=20)
    def get_parameter_array(self):
        return numpy.array([param.value for param in
                          self.circuitparameter_set.order_by('index')])

class CircuitParameter(models.Model):
    set = models.ForeignKey(CircuitParameterSet, on_delete=models.CASCADE)
    index = models.IntegerField(default=0)
    value = models.FloatField(default=0)



# params: {r, r_zarc_inductance, r_zarc_i...
# ... q_warburg, q_inductance
# ... w_c_inductance, w_c_zarc_i...
# ... phi_warburg, phi_zarc_i...
# ... phi_inductance, phi_zarc_inductance

class ShiftScaleParameters(models.Model):
    r_alpha = models.FloatField(default=0)
    w_alpha = models.FloatField(default=0)

    def display(self):
        return "(r_alpha={},w_alpha={})".format(self.r_alpha, self.w_alpha)

    def to_dict(self):
        return {'r_alpha':self.r_alpha, 'w_alpha':self.w_alpha}


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
            automatic_active_sample_disp,

        )
    def get_sample_array(self):
        return numpy.array([[samp.log_ang_freq,samp.real_part,samp.imag_part] for samp in
                            self.impedancesample_set.filter(active=True).order_by('log_ang_freq')])

    def any_active(self):
        return self.impedancesample_set.filter(active=True).exists()

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


class InverseModel(models.Model):
    logdir = models.CharField(max_length=100)
    kernel_size = models.IntegerField(default=7)
    conv_filters = models.IntegerField(default=16)
    num_conv = models.IntegerField(default=2)

    def display(self):
        return "(InvM {}, logdir {}, kernel_size {}, conv_filters {}, num_conv {})".format(
            self.id,
            self.logdir,
            self.kernel_size,
            self.conv_filters,
            self.num_conv)



class FitSpectrum(models.Model):
    active = models.BooleanField(default=True)
    def get_sample_array(self):
        return numpy.array([[samp.log_ang_freq,samp.real_part,samp.imag_part] for samp in
                            self.fitsample_set.order_by('log_ang_freq')])

class FitSample(models.Model):
    fit = models.ForeignKey(FitSpectrum, on_delete=models.CASCADE)
    log_ang_freq = models.FloatField()
    real_part = models.FloatField()
    imag_part = models.FloatField()

class ActivitySetting(models.Model):
    useless = models.BooleanField(default=True)


class ActivitySample(models.Model):
    setting = models.ForeignKey(ActivitySetting, on_delete=models.CASCADE)
    sample = models.ForeignKey(ImpedanceSample, on_delete=models.CASCADE)
    active = models.BooleanField(default=True)

class InverseModelResult(models.Model):
    spectrum=models.ForeignKey(
        EISSpectrum,
        on_delete=models.CASCADE
    )
    inv_model = models.ForeignKey(
        InverseModel,
        on_delete=models.CASCADE
    )

    inductance = models.BooleanField(default=False)
    zarc_inductance = models.BooleanField(default=False)
    num_zarcs = models.IntegerField(default=3)

    activity_setting = models.OneToOneField(
        ActivitySetting,
        on_delete=models.CASCADE
    )

    shift_scale_parameters = models.OneToOneField(
        ShiftScaleParameters,
        on_delete=models.CASCADE
    )
    circuit_parameters = models.OneToOneField(
        CircuitParameterSet,
        on_delete=models.CASCADE
    )

    fit_spectrum = models.OneToOneField(
        FitSpectrum,
        on_delete=models.CASCADE
    )

    def get_circuit_parameters_in_original_form(self):
        restored_params = restore_params(self.circuit_parameters.get_parameter_array(),
                                         self.shift_scale_parameters.to_dict())
        return deparameterized_params(restored_params)


    def get_normalized_sample_array(self):
        arr = self.spectrum.get_sample_array()
        log_freq, re_z, im_z = normalized_spectrum((arr[:, 0], arr[:, 1], arr[:, 2]),
                                                   params=self.shift_scale_parameters.to_dict())

        return numpy.stack((log_freq, re_z, im_z), axis=-1)






class FinetuneResult(models.Model):
    inv_model_result=models.ForeignKey(
        InverseModelResult,
        on_delete=models.CASCADE
    )

    learning_rate = models.FloatField(default=1e-3)
    nll_coeff = models.FloatField(default=1e-1)
    ordering_coeff = models.FloatField(default=.5)
    simplicity_coeff = models.FloatField(default=.1)
    sensible_phi_coeff = models.FloatField(default=1.)

    circuit_parameters = models.OneToOneField(
        CircuitParameterSet,
        on_delete=models.CASCADE
    )

    fit_spectrum = models.OneToOneField(
        FitSpectrum,
        on_delete=models.CASCADE
    )

    def get_circuit_parameters_in_original_form(self):
        restored_params = restore_params(self.circuit_parameters.get_parameter_array(),
                                         self.inv_model_result.shift_scale_parameters.to_dict())
        return deparameterized_params(restored_params)

    def display(self):
        return \
            'inv_model_result {} lr {} nll {}  ordering {} simplicity {} sensible phi {}'.format(
            self.inv_model_result,
            self.learning_rate,
            self.nll_coeff,
            self.ordering_coeff,
           self.simplicity_coeff,
           self.sensible_phi_coeff, )