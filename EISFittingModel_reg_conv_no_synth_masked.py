import tensorflow as tf
import os
import argparse
import random
import contextlib
import numpy
import math
import matplotlib.pyplot as plt
import pickle



"""
wider Conv Residual Block
"""
#TODO: always reset every layer to 0 where mask is 0
class ConvResBlock(tf.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1, dropout=0.2,
                 trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        super(ConvResBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.dropout = dropout
        self.filters = filters
        self.conv1 = tf.layers.Conv1D(
            filters=self.filters, kernel_size=kernel_size , strides=strides,
            dilation_rate=dilation_rate, activation=None, padding="same",
            name="conv1", kernel_initializer=tf.initializers.orthogonal)
        self.conv2 = tf.layers.Conv1D(
            filters=self.filters, kernel_size=kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=None, padding="same",
            name="conv2", kernel_initializer=tf.initializers.orthogonal)


        self.down_sample = None

    def build(self, input_shape_):
        input_shape = input_shape_[1]

        channel_dim = 2
        self.dropout1 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.filters)])
        self.dropout2 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.filters)])
        if input_shape[channel_dim] != self.filters:
            self.down_sample = tf.layers.Conv1D(
                 self.filters, kernel_size=1,
                 activation=None, data_format="channels_last", padding="same", kernel_initializer=tf.initializers.orthogonal, name="down_sample")

        self.built = True

    def call(self, inputs_, training=True, debug=False):

        # first channel of inputs is always going to be the masks
        masks = inputs_[0]
        masks_float = tf.cast(masks, dtype=tf.float32)
        inputs = inputs_[1]
        #clean the layer
        inputs = inputs * tf.expand_dims(masks_float, axis=2)

        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        # clean the layer
        x = x * tf.expand_dims(masks_float, axis=2)
        x = tf.layers.batch_normalization(x, trainable=training, renorm=True)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        # clean the layer
        x = x * tf.expand_dims(masks_float, axis=2)
        x = tf.layers.batch_normalization(x, trainable=training,gamma_initializer=tf.zeros_initializer(), renorm=True)
        x = self.dropout2(x, training=training)

        if self.down_sample is not None:
            inputs_down_sample = self.down_sample(inputs)
        else:
            inputs_down_sample = tf.identity(inputs)

        return tf.nn.relu(x + inputs_down_sample) * tf.expand_dims(masks_float, axis=2)




def Prior():
    '''

    :param frequencies:
    :return:
    '''




    mu = tf.constant([
        -1.,
        -2.,
        -2.,
        -3.,
        -3.,
        -7.,
        -9.,
        9.,
        -2.,
        2.,
        6.,
        1.,
        1.5,
        1.,
        1.,
        0.,
        0.
    ])


    log_square_sigma = 2.*tf.log(tf.constant([
        2.,
        2.,
        3.,
        3.,
        3.,
        4.,
        1.5,
        3.,
        5.,
        5.,
        5.,
        4.,
        3,
        3.,
        5.,
        0.01,
        0.01
    ]))


    return mu,log_square_sigma


#TODO: make some components maskable
def ImpedanceModel(params_, frequencies_, batch_size):
    '''

    :param params:
    :param frequencies:
    :return:
    '''

    params = tf.to_double(params_)
    frequencies = tf.to_double(frequencies_)
    # params: {r, r_zarc_inductance, r_zarc_i...
    # ... q_warburg, q_inductance
    # ... w_c_inductance, w_c_zarc_i...
    # ... phi_warburg, phi_zarc_i...
    # ... phi_inductance, phi_zarc_inductance
    number_of_zarc =3

    params_reshaped = tf.expand_dims(params, axis=2)

    batch_zeros = tf.zeros([batch_size, 1], dtype = tf.float64)
    full_zeros = tf.zeros_like(frequencies, dtype = tf.float64)


    first_mark =1+1+number_of_zarc+2+1+number_of_zarc
    second_mark = first_mark + 1 + number_of_zarc


    params_to_exp = tf.exp(params_reshaped[:, :first_mark])
    params_to_sigm = tf.sigmoid(params_reshaped[:, first_mark:second_mark])
    params_to_neg_sigm = -1./(1. + tf.square(params_reshaped[:, second_mark:]))

    '''
       - Resistor, parameters {R}: Z(W) = R + i * 0
       - Constant Phase Element, parameters {q, phi}: Z(W) = exp(q) * W^-Phi * (i)^-Phi 
       - Zarc, parameters {R, W_c, Phi}: Z(W) = R/(1 + (i W/W_c)^(Phi))


    '''
    exp_frequencies = tf.exp(frequencies)

    R = params_to_exp[:,0]


    impedance = tf.complex(full_zeros + R,full_zeros)

    imaginary_unit = tf.to_complex128(tf.complex(0.,1.))

    # warburg
    phi = params_to_sigm[:,0]
    exp_q = params_to_exp[:,5]

    bad_piece = tf.pow(imaginary_unit, tf.complex(-phi, batch_zeros))
    real_bad_piece = tf.real(bad_piece)
    imag_bad_piece = tf.imag(bad_piece)
    bad_piece2 = exp_q * tf.pow(exp_frequencies, -phi)

    impedance += tf.complex(
        real_bad_piece * bad_piece2, imag_bad_piece * bad_piece2
               )

    # inductance
    phi = params_to_neg_sigm[:,0]
    exp_q = params_to_exp[:, 6]

    bad_piece = tf.pow(imaginary_unit, tf.complex(-phi, batch_zeros))
    real_bad_piece = tf.real(bad_piece)
    imag_bad_piece = tf.imag(bad_piece)
    bad_piece2 = exp_q * tf.pow(exp_frequencies, -phi)

    impedance += tf.complex(
        real_bad_piece * bad_piece2, imag_bad_piece * bad_piece2
               )

    #inductance zarc
    phi = tf.complex(params_to_neg_sigm[:,1], batch_zeros)
    w_c = tf.complex(params_to_exp[:,7], batch_zeros)
    r = tf.complex(params_to_exp[:,1], batch_zeros)
    imag_freq = tf.complex(full_zeros, exp_frequencies)

    impedance += r/(1. + tf.pow((imag_freq/w_c), phi))


    for index in range(3):
        # zarc
        phi = tf.complex(params_to_sigm[:, 1+index], batch_zeros)
        w_c = tf.complex(params_to_exp[:, 8+index], batch_zeros)
        r = tf.complex(params_to_exp[:, 2+index], batch_zeros)

        impedance += r / (1. + tf.pow((imag_freq / w_c), phi))



    impedance_real = tf.real(impedance)
    impedance_imag = tf.imag(impedance)

    impedance_stacked = tf.to_float(tf.stack([impedance_real,impedance_imag], axis=2))

    return impedance_stacked

'''

def HardImpedanceModel(params_, masks, frequencies_, batch_size, number_of_zarcs=7):
    

    params = tf.to_double(params_)
    frequencies = tf.to_double(frequencies_)
    # params: {r, r_zarc_inductance, r_zarc_i...
    # ... q_warburg, q_inductance
    # ... w_c_inductance, w_c_zarc_i...
    # ... phi_warburg, phi_zarc_i...
    # ... phi_inductance, phi_zarc_inductance

    params_reshaped = tf.expand_dims(params, axis=2)

    batch_zeros = tf.zeros([batch_size, 1], dtype=tf.float64)
    full_zeros = tf.zeros_like(frequencies, dtype=tf.float64)

    first_mark = 1 + 1 + number_of_zarcs + 2 + 1 + number_of_zarcs
    second_mark = first_mark + 1 + number_of_zarcs

    params_to_exp = tf.exp(params_reshaped[:, :first_mark])
    params_to_sigm = tf.sigmoid(params_reshaped[:, first_mark:second_mark])
    params_to_neg_sigm = -1. / (1. + tf.square(params_reshaped[:, second_mark:]))

    

    last_frequencies = 1.0 + frequencies[:,-1]

    first_frequencies = -1.0 + frequencies[:,0]

    exp_frequencies = tf.exp(frequencies)

    R = params_to_exp[:, 0]

    impedance = tf.complex(full_zeros + R, full_zeros)

    imaginary_unit = tf.to_complex128(tf.complex(0., 1.))

    # warburg
    phi = params_to_sigm[:, 0]
    exp_q = params_to_exp[:, 2 + number_of_zarcs]

    bad_piece = tf.pow(imaginary_unit, tf.complex(-phi, batch_zeros))
    real_bad_piece = tf.real(bad_piece)
    imag_bad_piece = tf.imag(bad_piece)
    bad_piece2 = exp_q * tf.pow(exp_frequencies, -phi)

    impedance += tf.complex(
        real_bad_piece * bad_piece2, imag_bad_piece * bad_piece2
    )

    # inductance
    phi = params_to_neg_sigm[:, 0]
    exp_q = params_to_exp[:, 2 + number_of_zarcs + 1]

    bad_piece = tf.pow(imaginary_unit, tf.complex(-phi, batch_zeros))
    real_bad_piece = tf.real(bad_piece)
    imag_bad_piece = tf.imag(bad_piece)
    bad_piece2 = exp_q * tf.pow(exp_frequencies, -phi)

    impedance += tf.complex(
        real_bad_piece * bad_piece2, imag_bad_piece * bad_piece2
    )

    # inductance zarc
    phi = tf.complex(params_to_neg_sigm[:, 1], batch_zeros)
    w_c = tf.complex(params_to_exp[:, 2 + number_of_zarcs + 2 ], batch_zeros)
    r = tf.complex(params_to_exp[:, 1], batch_zeros)
    imag_freq = tf.complex(full_zeros, exp_frequencies)

    impedance += r / (1. + tf.pow((imag_freq / w_c), phi))

    for index in range(number_of_zarcs):
        # zarc
        phi = tf.complex(params_to_sigm[:, 1 + index], batch_zeros)
        w_c = tf.complex(params_to_exp[:, number_of_zarcs + 5 + index], batch_zeros)
        w_c_log = params[:, number_of_zarcs + 5 + index]
        freq_mask = tf.sigmoid(10.*(-w_c_log + last_frequencies)) * tf.sigmoid(10.*(w_c_log - first_frequencies))
        r = tf.complex( tf.expand_dims(freq_mask, axis=1) * masks[:,index:index+1]*params_to_exp[:, 2 + index], batch_zeros)

        impedance += r / (1. + tf.pow((imag_freq / w_c), phi))

    impedance_real = tf.real(impedance)
    impedance_imag = tf.imag(impedance)

    impedance_stacked = tf.to_float(tf.stack([impedance_real, impedance_imag], axis=2))

    return tf.stop_gradient(impedance_stacked)
'''

class ParameterVAE(object):

    def __init__(self, kernel_size, conv_filters, num_conv, trainable, num_encoded):
        self.kernel_size = kernel_size
        self.conv_filters = conv_filters

        self.num_conv = num_conv
        self.trainable = trainable
        self.num_encoded = num_encoded

        self.dropout = tf.placeholder(dtype=tf.float32)
        self.simplicity_coeff = tf.placeholder(dtype=tf.float32)
        self.nll_coeff = tf.placeholder(dtype=tf.float32)
        self.ordering_coeff = tf.placeholder(dtype=tf.float32)
        self.sensible_phi_coeff = tf.placeholder(dtype=tf.float32)

        self._input_layer = tf.layers.Conv1D(
            kernel_size=1, filters=self.conv_filters, strides=1,
            dilation_rate=1, activation=tf.nn.relu, trainable=trainable, data_format="channels_last",
            padding="valid",
            name="input_layer")

        self._input_layer_norm = tf.layers.BatchNormalization(scale=True, trainable=trainable, renorm=True,
                                                                    name="input_norm")



        self.encoding_layers = []
        for i in range(self.num_conv):
            if i == self.num_conv-1:
                filters = 2*self.conv_filters
            else:
                filters = self.conv_filters

            self.encoding_layers.append(
                ConvResBlock(filters=filters, kernel_size=self.kernel_size,
                             dropout=self.dropout, trainable=trainable,
                             name="conv_res_{}".format(i)))


        self._output_layer = tf.layers.Dense(
            units=self.num_encoded, activation=None, trainable=trainable,
            name="output_layer")

        self._output_layer_norm = tf.layers.BatchNormalization(scale=True, trainable=trainable, renorm=True,
                                                       gamma_initializer=tf.zeros_initializer(),
                                                              name="output_norm")

    def build_forward(self, inputs, masks, batch_size, priors):
        projected_inputs = self._input_layer_norm(self._input_layer(inputs))

        hidden =projected_inputs


        for i in range(len(self.encoding_layers)):
            hidden = self.encoding_layers[i]([masks, hidden])

        hidden_preweights = hidden[:,:,self.conv_filters:]
        hidden_values = hidden[:,:, :self.conv_filters]

        #hidden_preweights = tf.where(tf.tile(tf.expand_dims(masks, axis=2), multiples=[1,1,self.conv_filters]), hidden_preweights, -1e1 * tf.ones_like(hidden_preweights))
        hidden_weights = tf.nn.softmax(hidden_preweights, axis=1)
        hidden_weights = hidden_weights * tf.expand_dims(tf.cast(masks,dtype=tf.float32), axis=2)

        hidden= tf.reduce_sum(hidden_weights*hidden_values, axis=1, keepdims=False)/tf.reduce_sum(hidden_weights, axis=1, keepdims=False)

        representation = self._output_layer_norm(self._output_layer(hidden)) +tf.expand_dims(priors, axis=0)

        # get a single vector

        representation_mu = representation[:, :]


        z = representation_mu


        frequencies = inputs[:,:,0]
        impedances = ImpedanceModel(z, frequencies, batch_size=batch_size)


        return impedances, representation_mu




    def optimize_direct(self, inputs,
                        valid_freqs_counts,
                        freqs_num,
                        prior_mu, prior_log_sigma_sq,
                        learning_rate, global_norm_clip,
                        logdir, batch_size, trainable=True):

        masks_logical = tf.sequence_mask(
                            lengths=valid_freqs_counts,
                            maxlen=freqs_num,
                        )

        masks_float = tf.cast(masks_logical,dtype=tf.float32)

        impedances, representation_mu = self.build_forward(inputs, masks_logical, batch_size=batch_size, priors=prior_mu)

        _, variances = tf.nn.weighted_moments(inputs[:, :, 1:], axes=[1],
                                              frequency_weights=tf.expand_dims(masks_float, axis=2), keep_dims=False)
        std_devs = 1.0/(0.02 + tf.sqrt(variances))

        reconstruction_loss = (
            tf.reduce_sum(
                masks_float *
                    tf.reduce_mean(
                        tf.square(
                            tf.expand_dims(std_devs, axis=1) *
                            (impedances - inputs[:, :, 1:])
                        ),
                        axis=[2]
                    ),
                axis=[1]
            ) /
            tf.reduce_sum(masks_float, axis=[1])
        )


        # simplicity loss
        rs = representation_mu[:, 2:2 + 3]
        l_half = tf.square(tf.reduce_sum(tf.exp(.5 * rs), axis=1))
        l_1 = tf.reduce_sum(tf.exp(rs), axis=1)
        simplicity_loss = (l_half + l_1)
        complexity_metric = tf.reduce_mean(l_half / (1e-10 + l_1))

        # sensible_phi loss

        number_of_zarcs = 3

        first_mark = 1 + 1 + number_of_zarcs + 2 + 1 + number_of_zarcs + 1
        second_mark = first_mark + 1 + number_of_zarcs

        phi_warburg = tf.sigmoid(representation_mu[:, 1 + 1 + number_of_zarcs + 2 + 1 + number_of_zarcs])
        phi_zarcs = tf.sigmoid(representation_mu[:, first_mark:second_mark])

        sensible_phi_loss = (
            tf.square(tf.nn.relu(0.4 - phi_warburg)) +
            tf.square(tf.nn.relu(phi_warburg - 0.6)) +
            tf.nn.relu(0.5 - phi_zarcs[:, 1]) +
            tf.nn.relu(0.5 - phi_zarcs[:, 2]) +
            tf.nn.relu(0.5 - phi_zarcs[:, 3])
        )


        number_of_zarcs = 3
        first_wc_index = 2 + number_of_zarcs + 3
        wcs = representation_mu[:,
              first_wc_index:first_wc_index + number_of_zarcs]


        frequencies = inputs[:,:,0]
        max_frequencies =tf.gather_nd(
            params=frequencies,
            indices=tf.stack((tf.range(batch_size), valid_freqs_counts), axis=1)
        )

        ordering_loss = (
            tf.nn.relu(wcs[:, 0] - wcs[:, 1]) +
            tf.nn.relu(wcs[:, 1] - wcs[:, 2]) +
            tf.nn.relu(wcs[:, 2] - max_frequencies[:]) +
            tf.nn.relu(frequencies[:, 0] - wcs[:, 0])
        )

        prior_mu_ = tf.expand_dims(prior_mu, axis=0)
        prior_log_sigma_sq_ = tf.expand_dims(prior_log_sigma_sq, axis=0)

        nll_loss = \
         0.5 * tf.reduce_mean(
                tf.exp(- prior_log_sigma_sq_) * tf.square(representation_mu-prior_mu_),
             axis = 1
         )

        loss = tf.reduce_mean(
            tf.stop_gradient(reconstruction_loss)) * (
                    tf.reduce_mean(sensible_phi_loss) * self.sensible_phi_coeff +
                    tf.reduce_mean(nll_loss) * self.nll_coeff +
                    tf.reduce_mean(simplicity_loss) * self.simplicity_coeff +
                    tf.reduce_mean(ordering_loss) * self.ordering_coeff
            ) + tf.reduce_mean(reconstruction_loss)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        if trainable:
            with tf.name_scope('summaries'):
                tf.summary.scalar('loss', loss)
                tf.summary.scalar('nll_loss', tf.reduce_mean(nll_loss))
                tf.summary.scalar('simplicity loss', complexity_metric)
                tf.summary.scalar('ordering loss', tf.reduce_mean(ordering_loss))
                tf.summary.scalar('sensible phi loss', tf.reduce_mean(sensible_phi_loss))
                tf.summary.scalar('l1 loss', tf.reduce_mean(l_1))
                tf.summary.scalar('l1/2 loss', tf.reduce_mean(l_half))
                tf.summary.scalar('sqrt reconstruction_loss', tf.sqrt(reconstruction_loss))
                tf.summary.scalar('average mu', tf.reduce_mean(representation_mu))

            self.merger = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(os.path.join(logdir, 'train'))
            self.test_writer = tf.summary.FileWriter(os.path.join(logdir, 'test'))

            """
            we clip the gradient by global norm, currently the default is 10.
            -- Samuel B., 2018-09-14
            """
            optimizer = tf.train.AdamOptimizer(learning_rate)
            tvs = tf.trainable_variables()
            accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                gvs = optimizer.compute_gradients(loss, tvs)

            test_ops = tf.reduce_any(tf.concat([[tf.reduce_any(tf.is_nan(gv[0]), keepdims=False)] for i, gv in enumerate(gvs)],axis=0))

            accum_ops = tf.cond(test_ops, false_fn=lambda:[accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)], true_fn=lambda:[accum_vars[i].assign_add(tf.zeros_like(gv[0])) for i, gv in enumerate(gvs)])
            with tf.control_dependencies(accum_ops):
                gradients, _ = tf.clip_by_global_norm(accum_vars, global_norm_clip)
            train_step = optimizer.apply_gradients([(gradients[i], gv[1]) for i, gv in enumerate(gvs)])

            return loss, zero_ops, accum_ops, train_step, test_ops, impedances,  representation_mu, reconstruction_loss

        else:

            return loss, impedances, representation_mu, reconstruction_loss


class NonparametricOptimizer(object):
    """
    This is my implementation of Adam optimizer to be used on the EC parameters.

    """

    def __init__(self, parameter_matrix, spectrum_matrix, mask_matrix, extrema_freqs_matrix):

        self.sensible_phi_coeff = tf.placeholder(dtype=tf.float32)
        self.simplicity_coeff = tf.placeholder(dtype=tf.float32)
        self.nll_coeff = tf.placeholder(dtype=tf.float32)
        self.ordering_coeff = tf.placeholder(dtype=tf.float32)
        self.parameter_matrix = tf.get_variable(name='parameter_matrix', initializer=parameter_matrix)

        self.spectrum_matrix = tf.constant(value=spectrum_matrix,
                                           name='spectrum_matrix',
                                           shape=spectrum_matrix.shape,
                                            )
        self.mask_matrix = tf.constant(value=mask_matrix,
                                           name='mask_matrix',
                                           shape=mask_matrix.shape,
                                           )

        self.extrema_freqs_matrix = tf.constant(value=extrema_freqs_matrix,
                                       name='extrema_freqs_matrix',
                                       shape=extrema_freqs_matrix.shape,
                                       )

    def build_forward(self, frequencies, batch_size, indices):



        z = tf.gather_nd(
            params=self.parameter_matrix,
            indices=tf.expand_dims(indices, axis=1)
        )

        impedances = ImpedanceModel(z, frequencies, batch_size=batch_size)

        return impedances, z


    def optimize_direct(self,
                        indices, prior_mu,
                        prior_log_sigma_sq,
                        learning_rate,
                        batch_size, trainable=True):

        inputs = tf.gather_nd(
            params=self.spectrum_matrix,
            indices=tf.expand_dims(indices, axis=1)
        )

        extrema_freqs = tf.gather_nd(
            params=self.extrema_freqs_matrix,
            indices=tf.expand_dims(indices, axis=1)
        )

        impedances, representation_mu = self.build_forward(
            frequencies=inputs[:,:,0],
            batch_size=batch_size,
            indices=indices
        )

        masks = tf.gather_nd(
            params=self.mask_matrix,
            indices=tf.expand_dims(indices, axis=1)
        )

        _, variances = tf.nn.weighted_moments(inputs[:,:,1:],axes=[1],frequency_weights=tf.expand_dims(masks, axis=2), keep_dims=False)
        std_devs = 1.0/(0.02 + tf.sqrt(variances))

        reconstruction_loss = tf.reduce_sum(masks * tf.reduce_mean(tf.square(
            tf.expand_dims(std_devs, axis=1) * (impedances - inputs[:,:,1:]))
            , axis=[2]), axis=[1]) / tf.reduce_sum(masks, axis=[1])

        # simplicity loss
        rs = representation_mu[:, 2:2 + 3]
        l_half = tf.square(tf.reduce_sum(tf.exp(.5 * rs), axis=1))
        l_1 = tf.reduce_sum(tf.exp(rs), axis=1)
        simplicity_loss = tf.reduce_mean(l_half + l_1)
        complexity_metric = tf.reduce_mean(l_half / (1e-10 + l_1))

        # sensible_phi loss

        number_of_zarcs = 3

        first_mark = 1 + 1 + number_of_zarcs + 2 + 1 + number_of_zarcs + 1
        second_mark = first_mark + 1 + number_of_zarcs

        phi_warburg = tf.sigmoid(representation_mu[:, 1 + 1 + number_of_zarcs + 2 + 1 + number_of_zarcs])
        phi_zarcs = tf.sigmoid(representation_mu[:, first_mark:second_mark])

        sensible_phi_loss = tf.reduce_mean(
            tf.square(tf.nn.relu(0.4 - phi_warburg)) +
            tf.square(tf.nn.relu(phi_warburg - 0.6)) +
            tf.nn.relu(0.5 - phi_zarcs[:, 1]) +
            tf.nn.relu(0.5 - phi_zarcs[:, 2]) +
            tf.nn.relu(0.5 - phi_zarcs[:, 3])
        )


        number_of_zarcs = 3
        first_wc_index = 2 + number_of_zarcs + 3
        wcs = representation_mu[:, first_wc_index:first_wc_index + number_of_zarcs]

        ordering_loss = tf.reduce_mean(
            tf.nn.relu(wcs[:, 0] - wcs[:, 1]) +
            tf.nn.relu(wcs[:, 1] - wcs[:, 2]) +
            tf.nn.relu(wcs[:, 2] - extrema_freqs[:, 1]) +
            tf.nn.relu(extrema_freqs[:, 0] - wcs[:, 0])
        )
        prior_mu_ = tf.expand_dims(prior_mu, axis=0)
        prior_log_sigma_sq_ = tf.expand_dims(prior_log_sigma_sq, axis=0)

        nll_loss = \
         0.5 * tf.reduce_mean(
                tf.exp(- prior_log_sigma_sq_) * tf.square(representation_mu-prior_mu_))

        loss = tf.stop_gradient(reconstruction_loss) * (sensible_phi_loss * self.sensible_phi_coeff +nll_loss * self.nll_coeff + simplicity_loss * self.simplicity_coeff + ordering_loss * self.ordering_coeff) + reconstruction_loss
        if trainable:


            """
            we clip the gradient by global norm, currently the default is 10.
            -- Samuel B., 2018-09-14
            """
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_step = optimizer.minimize(tf.reduce_sum(loss))

            return tf.reduce_sum(loss), train_step,  impedances,  representation_mu, reconstruction_loss

        else:

            return tf.reduce_sum(loss), impedances, representation_mu, reconstruction_loss




    def get_indexed_matrices(
            self,
            indices,
            batch_size):

        inputs = tf.gather_nd(
            params=self.spectrum_matrix,
            indices=tf.expand_dims(indices, axis=1)
        )
        frequencies = inputs[:, :, 0]
        impedances, z = self.build_forward(
            frequencies=frequencies,
            batch_size=batch_size,
            indices=indices
        )

        masks = tf.gather_nd(
            params=self.mask_matrix,
            indices=tf.expand_dims(indices, axis=1)
        )

        return {
            'frequencies':  frequencies,
            'in_impedances':  inputs[:,:,1:],
            'out_impedances': impedances,
            'masks':  masks,
            'parameters':  z

        }









def finetune_test_data_with_adam(args):
    names_of_paths = {
        'fra': {
            'database': "database.file",
            'database_augmented': "database_augmented.file",
            'results': "results_of_inverse_model.file",
            'results_compressed': "results_compressed.file",
            'finetuned':"results_fine_tuned_with_adam_{}.file"
        },
        'eis': {
            'database': "database_eis.file",
            'database_augmented': "database_augmented_eis.file",
            'results': "results_of_inverse_model_eis.file",
            'results_compressed': "results_compressed_eis.file",
            'finetuned': "results_eis_fine_tuned_with_adam_{}.file"
        }

    }
    name_of_paths = names_of_paths[args.file_types]


    batch_size = tf.placeholder(dtype=tf.int32)
    prior_mu, prior_log_sigma_sq = Prior()
    frequencies = tf.placeholder(shape=[None, None], dtype=tf.float32)
    input_impedances = tf.placeholder(shape=[None, None, 2], dtype=tf.float32)

    number_of_zarcs = 3
    number_of_params = 1 + 1 + number_of_zarcs + 1 + 1 + 1 + number_of_zarcs + 1 + number_of_zarcs + 1 + 1
    params = tf.placeholder(shape=[None, number_of_params], dtype=tf.float32)

    m = tf.placeholder(shape=[None, number_of_params], dtype=tf.float32)
    v = tf.placeholder(shape=[None, number_of_params], dtype=tf.float32)
    model = NonparametricOptimizerAdam(num_encoded=number_of_params, beta1=args.adam_beta1, beta2=args.adam_beta2,
                                       epsilon=args.adam_epsilon)

    adam_time = tf.placeholder(dtype=tf.float32)
    loss, updates, updates_m,updates_v, impedances = \
        model.optimize_direct(
            current_params=params, current_m=m,current_v=v, frequencies=frequencies, in_impedances=input_impedances,  prior_mu=prior_mu,
                              prior_log_sigma_sq=prior_log_sigma_sq,
                              learning_rate=6e-3, batch_size=batch_size, adam_time = adam_time)



    if args.use_compressed:
        with open(os.path.join(".", "RealData", name_of_paths['results_compressed']), 'rb') as f:
            results = pickle.load(f)
    else:
        with open(os.path.join(".", "RealData", name_of_paths['results']), 'rb') as f:
            results = pickle.load(f)

    cleaned_data = sorted(results, key=lambda x: len(x[0]))

    grouped_data = []

    current_group = []

    max_batch_size = 5*128
    for freq, in_z, out_z, params_val, file_id in cleaned_data:
        current_len = len(freq)
        if len(current_group) == 0:
            current_group.append((freq, in_z, out_z, params_val, file_id))
        elif current_len == len(current_group[0][0]):
            current_group.append((freq, in_z, out_z, params_val, file_id))
            if len(current_group) == max_batch_size:
                grouped_data.append(copy.deepcopy(current_group))
                current_group = []
        else:
            grouped_data.append(copy.deepcopy(current_group))
            current_group = [(freq, in_z, out_z, params_val, file_id)]

    if not len(current_group) == 0:
        grouped_data.append(current_group)


    grouped_data_numpy = []
    for g in grouped_data:
        batch_len = len(g)
        batch_frequecies = numpy.array([x[0] for x in g])
        batch_in_impedances = numpy.array([x[1] for x in g])
        batch_out_impedances = numpy.array([x[2] for x in g])

        batch_params = numpy.array([x[3] for x in g])
        batch_file_ids = numpy.array([x[4] for x in g])
        # initialize to 0.
        batch_m = numpy.array([numpy.zeros(shape=number_of_params, dtype=numpy.float32) for _ in g])
        batch_v = numpy.array([numpy.zeros(shape=number_of_params, dtype=numpy.float32) for _ in g])

        grouped_data_numpy.append((batch_len, batch_frequecies, batch_in_impedances, batch_out_impedances, batch_params, batch_m,batch_v, batch_file_ids))


    with tf.Session() as sess:
        for j in range(args.total_steps):
            total_loss = 0.0
            for i in range(len(grouped_data_numpy)):
                batch_len, batch_frequecies, batch_in_impedances, _, batch_params, batch_m,batch_v, batch_file_ids = grouped_data_numpy[i]



                loss_value, out_impedance, new_params_value, new_m_values, new_v_values = \
                    sess.run(
                        [ loss, impedances, updates, updates_m, updates_v],
                        feed_dict={batch_size: batch_len,
                                   model.sensible_phi_coeff: args.sensible_phi_coeff,
                                   model.simplicity_coeff: args.simplicity_coeff,
                                   model.nll_coeff: args.nll_coeff,
                                   model.ordering_coeff: args.ordering_coeff,
                                   frequencies: batch_frequecies,
                                   input_impedances: batch_in_impedances,
                                   params: batch_params,
                                   m:batch_m,
                                   v:batch_v,
                                   adam_time:float(j+1)
                                   })
                total_loss += loss_value
                grouped_data_numpy[i] = (batch_len, batch_frequecies, batch_in_impedances, out_impedance, new_params_value,new_m_values, new_v_values, batch_file_ids)

            print('iteration {}, total loss {}.'.format(j, total_loss))

            if j % args.log_every == 0:
                new_results = []
                for i in range(len(grouped_data_numpy)):
                    batch_len, batch_frequecies, batch_in_impedances, batch_out_impedances, batch_params, _, _, batch_file_ids= \
                        grouped_data_numpy[i]


                    for k in range(len(batch_frequecies)):
                        new_results.append((batch_frequecies[k], batch_in_impedances[k],batch_out_impedances[k], batch_params[k], batch_file_ids[k]))

                with open(os.path.join(".", "RealData", name_of_paths['finetuned'].format(j)), 'wb') as f:
                    pickle.dump(new_results, f, pickle.HIGHEST_PROTOCOL)







@contextlib.contextmanager
def initialize_session(logdir, sampling =False, seed=None):
    """Create a session and saver initialized from a checkpoint if found."""
    numpy.random.seed(seed=seed)

    if sampling:
        config = tf.ConfigProto(
            #device_count={'GPU': 0}
        )
    else:
        config = tf.ConfigProto(
            #device_count={'GPU': 0}
        )
    # config.gpu_options.allow_growth = True
    logdir = os.path.expanduser(logdir)
    checkpoint = tf.train.latest_checkpoint(logdir)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        if checkpoint:
            print('Load checkpoint {}.'.format(checkpoint))
            saver.restore(sess, checkpoint)
        else:
            print('Initialize new model.')
            os.makedirs(logdir, exist_ok=True)
            sess.run(tf.global_variables_initializer())
        yield sess, saver

















class GroupBy():
    def __init__(self):
        self.data = {}

    def record(self, k, v):
        if k in self.data.keys():
            self.data[k].append(v)
        else:
            self.data[k] = [v]


import copy

class GetFresh:
    """
    Get fresh numbers, either
        - from 0 to n_samples-1 or
        - from list_of_indecies
    in a random order without repetition
    However, once we have exausted all the numbers, we reset.
    - Samuel Buteau, October 2018
    """

    def __init__(self, n_samples=None, list_of_indecies=None):
        if not n_samples is None:
            self.GetFresh_list = numpy.arange(n_samples, dtype=numpy.int32)
            self.get_fresh_count = n_samples
        elif not list_of_indecies is None:
            self.GetFresh_list = numpy.array(copy.deepcopy(list_of_indecies))
            self.get_fresh_count = len(self.GetFresh_list)
        else:
            raise Exception('Invalid Input')

        numpy.random.shuffle(self.GetFresh_list)
        self.get_fresh_pos = 0

    def get(self, n):
        """
        will return a list of n random numbers in self.GetFresh_list
        - Samuel Buteau, October 2018
        """
        if n >= self.get_fresh_count:
            return self.GetFresh_list

        reshuffle_flag = False

        n_immediate_fulfill = min(n, self.get_fresh_count - self.get_fresh_pos)
        batch_of_indecies = numpy.empty([n], dtype=numpy.int32)
        for i in range(0, n_immediate_fulfill):
            batch_of_indecies[i] = self.GetFresh_list[i + self.get_fresh_pos]

        self.get_fresh_pos += n_immediate_fulfill
        if self.get_fresh_pos >= self.get_fresh_count:
            self.get_fresh_pos -= self.get_fresh_count
            reshuffle_flag = True

            # Now, the orders that needed to be satisfied are satisfied.
        n_delayed_fulfill = max(0, n - n_immediate_fulfill)
        if reshuffle_flag:
            numpy.random.shuffle(self.GetFresh_list)

        if n_delayed_fulfill > 0:
            for i in range(0, n_delayed_fulfill):
                batch_of_indecies[i + n_immediate_fulfill] = self.GetFresh_list[i]
            self.get_fresh_pos = n_delayed_fulfill

        return batch_of_indecies







def split_train_test_data(args, file_types=None):
    names = {'fra':{'database':"database.file",'database_split':"database_split_{}.file"},
             'eis':{'database':"database_eis.file",'database_split':"database_eis_split_{}.file"}}

    if file_types is None:
        file_types = args.file_types

    name = names[file_types]
    if not os.path.isfile(os.path.join(".", "RealData", name['database_split'].format(
            args.percent_training))):
        with open(os.path.join(".", "RealData", name['database']), 'rb') as f:
            database = pickle.load(f)

        # this is where we split in test and train.


        if file_types == 'eis':
            all_keys = list(database.keys())
            random.shuffle(all_keys)

            total_count = len(all_keys)

            train_count = int(float(total_count) * float(args.percent_training)/100.)
            train_keys = all_keys[:train_count]
            test_keys = all_keys[train_count:]

            split = {}
            for train_key in train_keys:
                split[train_key] = {'train':True}
            for test_key in test_keys:
                split[test_key] = {'train':False}

            with open(os.path.join(".", "RealData", name['database_split'].format(
                    args.percent_training)), 'wb') as f:
                 pickle.dump(split, f, pickle.HIGHEST_PROTOCOL)

        elif file_types == 'fra':
            cell_id_groups = get_cell_id_groups(database)
            all_keys = list(cell_id_groups.keys())
            random.shuffle(all_keys)

            total_count = len(all_keys)

            train_count = int(float(total_count) * float(args.percent_training) / 100.)
            train_keys = all_keys[:train_count]
            test_keys = all_keys[train_count:]

            split = {}
            for train_cell in train_keys:
                for file_id in cell_id_groups[train_cell]:
                    split[file_id] = {'train': True}

            for test_cell in test_keys:
                for file_id in cell_id_groups[test_cell]:
                    split[file_id] = {'train': False}

            with open(os.path.join(".", "RealData", name['database_split'].format(
                    args.percent_training)), 'wb') as f:
                pickle.dump(split, f, pickle.HIGHEST_PROTOCOL)

    else:
        with open(os.path.join(".", "RealData", name['database_split'].format(
                args.percent_training)), 'rb') as f:
            split = pickle.load(f)

    return split

def train_on_all_data(args):
    print('args.new_logdir = {}.'.format(args.new_logdir))
    if args.new_logdir:
        logdir_new = args.logdir + 'NEW_Fast'
    else:
        logdir_new = args.logdir

    random.seed(a=args.seed)
    number_of_zarcs = 3
    number_of_params = 1 + 1 + number_of_zarcs + 1 + 1 + 1 + number_of_zarcs + 1 + number_of_zarcs + 1 + 1
    batch_size = args.batch_size
    prior_mu, prior_log_sigma_sq = Prior()


    split = split_train_test_data(args, file_types='fra')
    split_eis = split_train_test_data(args, file_types='eis')

    with open(os.path.join(".", "RealData", "database.file"), 'rb') as f:
        data = pickle.load(f)

    cleaned_data = []
    for file_id in data.keys():
        if not split[file_id]['train']:
            continue

        log_freq, re_z, im_z = data[file_id]['original_spectrum']
        negs = data[file_id]['freqs_with_negative_im_z']
        n_freq = len(log_freq)
        if len(log_freq) < 10:
            continue

        for i_negs in range(min(len(log_freq), negs+5)):
            log_freq_negs = log_freq[:n_freq-i_negs]
            re_z_negs = re_z[:n_freq-i_negs]
            im_z_negs = im_z[:n_freq-i_negs]

            # here, cleaned_data doesn't need to remember the database id.
            cleaned_data.append(copy.deepcopy((log_freq_negs, re_z_negs, im_z_negs)))


    cleaned_data_lens = [len(c[0]) for c in cleaned_data]


    with open(os.path.join(".", "RealData", "database_eis.file"), 'rb') as f:
        data_eis = pickle.load(f)

    cleaned_data_eis = []
    for file_id in data_eis.keys():
        if not split_eis[file_id]['train']:
            continue

        log_freq, re_z, im_z = data_eis[file_id]['original_spectrum']
        negs = data_eis[file_id]['freqs_with_negative_im_z']
        tails = data_eis[file_id]['freqs_with_tails_im_z']
        n_freq = len(log_freq)
        if len(log_freq) < 10:
            continue

        lower_bound = min(len(log_freq), max(tails,negs)+5)
        upper_bound = min([max(tails,negs), len(log_freq) -1 , 7])
        for i_negs in range(upper_bound, lower_bound):
            log_freq_negs = log_freq[:n_freq-i_negs]
            re_z_negs = re_z[:n_freq-i_negs]
            im_z_negs = im_z[:n_freq-i_negs]

            # here, cleaned_data doesn't need to remember the database id.
            cleaned_data_eis.append(copy.deepcopy((log_freq_negs, re_z_negs, im_z_negs)))


    cleaned_data_lens_eis = [len(c[0]) for c in cleaned_data_eis]

    max_freq_num = max(numpy.max(cleaned_data_lens),numpy.max(cleaned_data_lens_eis))


    spectrum_count = len(cleaned_data_lens)

    full_data_fra = numpy.zeros(shape=(spectrum_count,max_freq_num,3), dtype=numpy.float32)
    full_data_freqs_counts = numpy.zeros(shape=(spectrum_count), dtype=numpy.int32)



    for ind in range(spectrum_count):
        full_data_freqs_counts[ind] = len(cleaned_data[ind][0])
        for j in range(3):
            full_data_fra[ind,:full_data_freqs_counts[ind],j] = numpy.array(cleaned_data[ind][j])

    dataset_fra = tf.data.Dataset.from_tensor_slices((full_data_fra, full_data_freqs_counts))


    spectrum_count_eis = len(cleaned_data_lens_eis)

    full_data_eis = numpy.zeros(shape=(spectrum_count_eis, max_freq_num, 3), dtype=numpy.float32)
    full_data_freqs_counts_eis = numpy.zeros(shape=(spectrum_count_eis), dtype=numpy.int32)
    for ind in range(spectrum_count_eis):
        full_data_freqs_counts_eis[ind] = len(cleaned_data_eis[ind][0])
        for j in range(3):
            full_data_eis[ind, :full_data_freqs_counts_eis[ind], j] = numpy.array(cleaned_data_eis[ind][j])

    dataset_eis = tf.data.Dataset.from_tensor_slices((full_data_eis, full_data_freqs_counts_eis))

    dataset= tf.data.experimental.sample_from_datasets(
        datasets=(
            dataset_fra.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10000)),
            dataset_eis.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10000))
        ),
    )

    dataset= dataset.batch(args.batch_size)
    iterator = dataset.make_one_shot_iterator()
    inputs_fresh, valid_freqs_counts = iterator.get_next()


    frequencies = inputs_fresh[:,:,0]

    input_impedances = inputs_fresh[:,:,1:]






    epsilon_scale = .1 * tf.random_uniform(shape=[batch_size], minval=-2., maxval=2., dtype=tf.float32)
    epsilon_frequency_translate = .5 * tf.random_uniform(shape=[batch_size], minval=-2., maxval=2., dtype=tf.float32)

    squared_impedances = input_impedances[:, :, 0] ** 2 + input_impedances[:, :, 1] ** 2

    masks_logical = tf.sequence_mask(
        lengths=valid_freqs_counts,
        maxlen=max_freq_num,
    )

    masked_squared_impedances = tf.where(masks_logical, squared_impedances, tf.ones_like(squared_impedances)*(-tf.float32.max) )

    maxes = epsilon_scale -0.5 * tf.log(0.00001 + tf.reduce_max(masked_squared_impedances, axis=1))
    pure_impedances = tf.exp(tf.expand_dims(tf.expand_dims(maxes, axis=1), axis=2)) * input_impedances

    max_frequencies = tf.gather_nd(
        params=frequencies,
        indices=tf.stack((tf.range(batch_size), valid_freqs_counts), axis=1)
    )

    avg_freq = .5 * (frequencies[:, 0] + max_frequencies) + epsilon_frequency_translate

    pure_frequencies = frequencies - tf.expand_dims(avg_freq, axis=1)
    inputs = tf.concat([tf.expand_dims(pure_frequencies, axis=2), pure_impedances], axis=2)

    model = ParameterVAE(kernel_size=args.kernel_size, conv_filters=args.conv_filters, num_conv=args.num_conv,
                         trainable=True, num_encoded=number_of_params)

    loss, zero_ops, accum_ops, train_step, test_ops, impedances, representation_mu, my_reconstruction_loss = \
        model.optimize_direct(inputs=inputs,
                              valid_freqs_counts=valid_freqs_counts,
                              freqs_num=max_freq_num,
                              prior_mu=prior_mu,
                              prior_log_sigma_sq=prior_log_sigma_sq,
                              learning_rate=args.learning_rate,
                              global_norm_clip=args.global_norm_clip,
                              logdir=logdir_new, batch_size=batch_size)

    step = tf.train.get_or_create_global_step()
    increment_step = step.assign_add(1)

    reconstruction_loss_avg = 1.0




    with initialize_session(args.logdir, seed=args.seed) as (sess, saver):





        while True:
            current_step = sess.run(step)
            if current_step >= args.total_steps:
                print('Training complete.')
                break

            sess.run(zero_ops)
            summaries = []
            total_loss = 0.0

            '''
            nll_coeff = 10. * args.nll_coeff * 1./(1. + float(current_step))**.5
            simplicity_coeff =  args.simplicity_coeff /(1. + float(current_step))**(1./3.)
            ordering_coeff = .1 * args.ordering_coeff * (1. + float(current_step))**.5
            '''

            for count in range(args.virtual_batches):




                if count < args.virtual_batches - 1:
                    summary, reconstruction_loss_value, loss_value, _, test = \
                        sess.run([model.merger, my_reconstruction_loss, loss, accum_ops, test_ops],
                                 feed_dict={model.dropout: args.dropout,
                                            model.sensible_phi_coeff: args.sensible_phi_coeff,
                                            model.simplicity_coeff: args.simplicity_coeff,
                                            model.nll_coeff: args.nll_coeff,
                                            model.ordering_coeff:  args.ordering_coeff})


                else:

                    summary, reconstruction_loss_value, loss_value, _, test, step_value, freq, in_impedance, out_impedance = \
                        sess.run([model.merger, my_reconstruction_loss, loss, train_step, test_ops, increment_step ,pure_frequencies, pure_impedances, impedances],
                                 feed_dict={
                                            model.dropout: args.dropout,
                                            model.sensible_phi_coeff: args.sensible_phi_coeff,
                                            model.simplicity_coeff: args.simplicity_coeff,
                                            model.nll_coeff: args.nll_coeff,
                                            model.ordering_coeff:  args.ordering_coeff})


                reconstruction_loss_avg = reconstruction_loss_avg * .99 + reconstruction_loss_value * (1. - .99)

                summaries.append(summary)
                total_loss += loss_value

            total_loss /= float(args.virtual_batches)

            if not math.isfinite(total_loss):
                print('was not finite')
                # sess.run(tf.global_variables_initializer())
                # sess.run(zero_ops)
                # print('restarted')
                # continue

            if step_value % args.log_every == 0:
                print(
                    'Step {} loss {}, reconstruction_loss {}.'.format(step_value, total_loss, reconstruction_loss_avg))
                for summary in summaries:
                    model.train_writer.add_summary(summary, step_value)

            if step_value % args.checkpoint_every == 0:
                print('Saving checkpoint.')
                saver.save(sess, os.path.join(logdir_new, 'model.ckpt'), step_value)















def high_frequency_remove(spectrum, negs, tails=None):
    log_freq, re_z, im_z = spectrum

    negs_remove = int(1.5 * float(negs) / 4.0)

    if tails is None:
        tails_remove = 0
    else:
        tails_remove = max(0, tails)

    remove = max(negs_remove, tails_remove)
    n_freq = len(log_freq)
    log_freq = log_freq[:n_freq - remove]
    re_z = re_z[:n_freq - remove]
    im_z = im_z[:n_freq - remove]
    return (log_freq, re_z, im_z)


def shift_scale_param_extract(spectrum):

    log_freq, re_z, im_z = spectrum
    squared_impedances = re_z ** 2 + im_z ** 2
    r_alpha_from_unity = 0.5 * math.log(0.00001 + numpy.max(squared_impedances))
    w_alpha_from_unity = 0.5 * (log_freq[0] + log_freq[-1])
    return {'r_alpha': r_alpha_from_unity, 'w_alpha':w_alpha_from_unity}




def normalized_spectrum(spectrum, params):
    log_freq, re_z, im_z = spectrum

    unity_re_z = numpy.exp(-params['r_alpha']) * re_z
    unity_im_z = numpy.exp(-params['r_alpha']) * im_z

    unity_log_freq = - params['w_alpha'] + log_freq

    return (unity_log_freq, unity_re_z, unity_im_z)

def original_spectrum(spectrum, params):
    log_freq, re_z, im_z = spectrum

    original_re_z = numpy.exp(params['r_alpha']) * re_z
    original_im_z = numpy.exp(params['r_alpha']) * im_z

    original_log_freq =  params['w_alpha'] + log_freq

    return (original_log_freq, original_re_z, original_im_z)


def restore_params(params, shift_scale):
    new_params = copy.deepcopy(params)
    for i in range(5):
        new_params[i] = new_params[i] + shift_scale['r_alpha']

    new_params[5] = new_params[5] + shift_scale['r_alpha'] + (1./(1. + math.exp(-new_params[11]))) * shift_scale['w_alpha']
    new_params[6] = new_params[6] + shift_scale['r_alpha'] - (1./(1. + (new_params[15])**2.)) * shift_scale['w_alpha']
    for i in range(4):
        new_params[7+i] = new_params[7+i] + shift_scale['w_alpha']

    return new_params

list_of_labels = [
        'r_ohm',
    'r_zarc_inductance',
    'r_zarc_1', 'r_zarc_2', 'r_zarc_3',
    'q_warburg',
    'q_inductance',
    'w_c_inductance',
     'w_c_zarc_1', 'w_c_zarc_2', 'w_c_zarc_3',
    'phi_warburg',
    'phi_zarc_1', 'phi_zarc_2', 'phi_zarc_3',
    'phi_inductance',
    'phi_zarc_inductance'
    ]

def deparameterized_params(params):



    new_params = copy.deepcopy(params)
    for i in range(7):
        new_params[i] = math.exp(new_params[i])

    number_of_zarc = 3
    first_mark = 1 + 1 + number_of_zarc + 2 + 1 + number_of_zarc
    second_mark = first_mark + 1 + number_of_zarc

    for i in range(first_mark,second_mark):
        new_params[i] = 1./(1. + math.exp(-new_params[i]))

    for i in range(second_mark, len(list_of_labels)):

        new_params[i] = -1./(1. + (new_params[i])**2.)

    return new_params



def run_on_real_data(args):
    names_of_paths = {
        'fra':{
            'database':"database.file",
            'database_augmented':"database_augmented.file",
            'results':"results_of_inverse_model.file"
        },
        'eis':{
            'database': "database_eis.file",
            'database_augmented': "database_augmented_eis.file",
            'results': "results_of_inverse_model_eis.file"
        }

    }
    name_of_paths= names_of_paths[args.file_types]

    random.seed(a=args.seed)
    batch_size = tf.placeholder(dtype=tf.int32)
    prior_mu, prior_log_sigma_sq = Prior()

    frequencies = tf.placeholder(shape=[None, None], dtype=tf.float32)
    input_impedances = tf.placeholder(shape=[None, None,2], dtype=tf.float32)

    inputs = tf.concat([tf.expand_dims(frequencies, axis=2), input_impedances], axis=2)

    number_of_zarcs = 3
    number_of_params = 1 + 1 + number_of_zarcs + 1 + 1 + 1 + number_of_zarcs + 1 + number_of_zarcs + 1 + 1

    model = ParameterVAE(kernel_size=args.kernel_size, conv_filters=args.conv_filters,
                          num_conv=args.num_conv, trainable=False, num_encoded=number_of_params)

    loss, impedances, representation_mu, my_reconstruction_loss = \
        model.optimize_direct( inputs=inputs,prior_mu=prior_mu,
                                  prior_log_sigma_sq=prior_log_sigma_sq,
                                  learning_rate=args.learning_rate,
                                  global_norm_clip=args.global_norm_clip,
                                  logdir=args.logdir, batch_size=batch_size, trainable=False)

    split = split_train_test_data(args)
    with open(os.path.join(".", "RealData", name_of_paths['database']), 'rb') as f:
        data= pickle.load(f)

    cleaned_data = []

    for file_id in data.keys():
        if file_id in split.keys() and split[file_id]['train']:
            continue

        tails = None
        if 'freqs_with_tails_im_z' in data[file_id].keys():
            tails = data[file_id]['freqs_with_tails_im_z']
        cropped_spectrum = high_frequency_remove(spectrum=data[file_id]['original_spectrum'],
                                                 negs=data[file_id]['freqs_with_negative_im_z'],
                                                 tails=tails)

        shift_scale_params = shift_scale_param_extract(cropped_spectrum)

        data[file_id]['shift_scale_params'] = shift_scale_params
        log_freq, re_z, im_z = normalized_spectrum(cropped_spectrum, params=shift_scale_params)

        cleaned_data.append((log_freq,re_z,im_z, file_id))


    with open(os.path.join(".", "RealData", name_of_paths['database_augmented']), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    cleaned_data = sorted(cleaned_data, key=lambda x: len(x[0]))

    grouped_data = []

    current_group =[]
    for freq, re_z, im_z, file_id in cleaned_data:
        current_len = len(freq)
        if len(current_group) == 0:
            current_group.append((freq, re_z, im_z, file_id ))
        elif current_len == len(current_group[0][0]):
            current_group.append((freq, re_z,im_z, file_id ))
            if len(current_group) == 64:
                grouped_data.append(copy.deepcopy(current_group))
                current_group = []
        else:
            grouped_data.append(copy.deepcopy(current_group))
            current_group = [(freq, re_z, im_z, file_id )]

    if not len(current_group) == 0:
        grouped_data.append(current_group)



    results = []
    with initialize_session(logdir=args.logdir, seed=args.seed) as (sess, saver):
        for g in grouped_data:
            batch_len = len(g)
            batch_frequecies = numpy.array([x[0] for x in g])
            batch_impedances =  numpy.array([numpy.stack((x[1], x[2]), axis=1) for x in g])
            batch_file_ids = numpy.array([x[3] for x in g])

            reconstruction_loss_value, loss_value,out_impedance,in_impedance, freqs, representation_mu_value  = \
                sess.run([ my_reconstruction_loss,loss, impedances, input_impedances, frequencies, representation_mu],
                         feed_dict={batch_size: batch_len,
                                    model.dropout: 0.0,
                                    model.sensible_phi_coeff: args.sensible_phi_coeff,
                                    model.simplicity_coeff: args.simplicity_coeff,
                                    model.nll_coeff: args.nll_coeff,
                                    model.ordering_coeff: args.ordering_coeff,
                                    frequencies: batch_frequecies,
                                    input_impedances: batch_impedances
                                    })


            current_results = [(freqs[index], in_impedance[index],out_impedance[index], representation_mu_value[index], batch_file_ids[index]) for index in range(batch_len)]
            results += copy.deepcopy(current_results)


        with open(os.path.join(".", "RealData", name_of_paths['results']), 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)













def real_score(in_imp, fit_imp):
    real_scale = 1. / (0.0001 + numpy.std(in_imp[:, 0]))
    imag_scale = 1. / (0.0001 + numpy.std(in_imp[:, 1]))
    return (numpy.mean(
        (numpy.expand_dims(numpy.array([real_scale, imag_scale]), axis=0) * (in_imp - fit_imp)) ** 2.)) ** (1. / 2.)

def complexity_score(params):
    num_zarcs = 3
    rs = params[2:2 + num_zarcs]

    l_half = numpy.square(numpy.sum(numpy.exp(.5 * rs)))
    l_1 = numpy.sum(numpy.exp(rs))
    complexity_loss = l_half / (1e-8 + l_1)

    return complexity_loss

def plot_to_scale(args):
    with open(os.path.join(".", "RealData", "results_fine_tuned_with_adam_{}.file".format(args.plot_step)), 'rb') as f:
        results = pickle.load(f)

    with open(os.path.join(".", "RealData", "database_augmented.file"), 'rb') as f:
        database = pickle.load(f)



    sorted_sorted_results = sorted(results, key=lambda x: database[x[-1]]['shift_scale_params']['r_alpha'], reverse=True)

    list_to_print = sorted_sorted_results

    list_of_indecies = [[8,13,17,42],[100, 200,502,5001],[10000, 20002, 30000,40000],[ 45001,49501,49852,49997]]#[0.00005*float(x) for x in range(30)] + [0.004, 0.01, 0.1, 0.4, 0.9,0.98, 0.99,0.992,0.996,0.997,0.998] + [0.999 + 0.0001*float(x) for x in range(10)]
    fig = plt.figure()
    for i in range(len(list_of_indecies)):
        #i = int(i_frac * len(list_to_print))

        ax = fig.add_subplot(2,2,i+1)
        for j in list_of_indecies[i]:

            measured_log_freq = list_to_print[j][0][:]
            measured_re_z = list_to_print[j][1][:, 0]
            measured_im_z = list_to_print[j][1][:, 1]

            fitted_log_freq = list_to_print[j][0][:]
            fitted_re_z = list_to_print[j][2][:, 0]
            fitted_im_z = list_to_print[j][2][:, 1]

            fitted_params = list_to_print[j][3][:]
            c_score = complexity_score(fitted_params)
            e_score = real_score(list_to_print[j][1], list_to_print[j][2])

            file_id = list_to_print[j][4]
            shift_scale_params = database[file_id]['shift_scale_params']

            measured_rescaled_log_freq, measured_rescaled_re_z, measured_rescaled_im_z = \
                original_spectrum((measured_log_freq, measured_re_z, measured_im_z), shift_scale_params)

            fitted_rescaled_log_freq, fitted_rescaled_re_z, fitted_rescaled_im_z = \
                original_spectrum((fitted_log_freq, fitted_re_z, fitted_im_z), shift_scale_params)

            label = 'Error:{:1.3f}, Complexity:{:1.1f}'.format(e_score, c_score)
            print(label, 'index {}'.format(j))
            ax.scatter(measured_rescaled_re_z, -measured_rescaled_im_z )
            ax.plot(fitted_rescaled_re_z, -fitted_rescaled_im_z ,label=label )

        print('i: {}'.format(i))
        print('percentage: {}'.format(100. * float(i) / float(len(sorted_sorted_results))))
        ax.legend()
    plt.show()




def plot_param_histo(args):
    with open(os.path.join(".", "RealData", args.histogram_file), 'rb') as f:
        results = pickle.load(f)


    params = numpy.array(list(map(lambda x: x[3], results)))
    list_of_labels = ['r_ohm', 'r_zarc_inductance', 'r_zarc_1', 'r_zarc_2', 'r_zarc_3', 'q_warburg',
                      'q_inductance',
                      'w_c_inductance',
                      'w_c_zarc_1', 'w_c_zarc_2', 'w_c_zarc_3',
                      'phi_warburg',
                      'phi_zarc_1', 'phi_zarc_2', 'phi_zarc_3',
                      'phi_inductance',
                      'phi_zarc_inductance'
                      ]

    ones = 1.
    zeros = 0.
    number_of_zarc = 3
    list_of_priors = (1 + 1 + number_of_zarc) * [.5 * (math.log(0.0001) + math.log(1.)) * ones] + [
        .5 * (math.log(0.001) + math.log(.00001)) * ones,
        .5 * (math.log(0.001) - 2. * math.log(10000000.)) * ones,
        (math.log(10) + math.log(10000000.)) * ones,
        .5 * (math.log(.001)) * ones,
        zeros,
        .5 * (math.log(1000.)) * ones,
        .5 * (-math.log(1. / .75 - 1.) - math.log(1. / .4 - 1.)) * ones] + \
    number_of_zarc * [.5 * (-math.log(1. / .95 - 1.) - math.log(1. / .6 - 1.)) * ones] + \
    2 * [zeros]

    for i in range(len(params[0])):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(params[:, i], bins=100)
        plt.xlabel(list_of_labels[i] + ',   prior: {:2.5f}'.format(list_of_priors[i]))
        plt.show()



def get_cell_id_groups(database):
    metadata_groups = {}

    for file_id in database.keys():
        cell_id = database[file_id]['cell_id']
        if not cell_id in metadata_groups.keys():
            metadata_groups[cell_id] = []
        metadata_groups[cell_id].append(file_id)

    return metadata_groups


def tally_spectrum_lengths(args):
    #TODO
    return


def compress_data(args):
    with open(os.path.join(".", args.data_dir, "results_of_inverse_model.file"), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(".", args.data_dir, "database_augmented.file"), 'rb') as f:
        database = pickle.load(f)



    sorted_sorted_results = sorted(results, key=lambda x: real_score(x[1], x[2]), reverse=True)

    metadata_groups = {}

    for res_i in range(len(sorted_sorted_results)):
        res = sorted_sorted_results[res_i]
        file_id = res[-1]
        meta = database[file_id]
        cell_id = meta['cell_id']
        if not cell_id in metadata_groups.keys():
            metadata_groups[cell_id] = []

        metadata_groups[cell_id].append(
            {'file_id': file_id, 'index': res_i})

    random_keys = list(metadata_groups.keys())
    random.shuffle(random_keys)

    count = 0

    compressed_result = []
    compressed_n = args.compressed_num
    compressed_ids = []
    for r_k in random_keys:
        my_meta = metadata_groups[r_k]
        count += len(my_meta)
        if count > compressed_n:
            break

        print('adding {} spectra'.format(len(my_meta)))
        for di in my_meta:
            compressed_result.append(sorted_sorted_results[di['index']])
            compressed_ids.append(di['file_id'])

    compressed_database = {}
    for id in compressed_ids:
        compressed_database[id] = database[id]

    with open(os.path.join(".", args.data_dir, "results_compressed.file"), 'wb') as f:
        pickle.dump(compressed_result, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(".", args.data_dir, "database_compressed.file"), 'wb') as f:
        pickle.dump(compressed_database, f, pickle.HIGHEST_PROTOCOL)


    '''
    compressed_result = []
    compressed_n = args.compressed_num
    start = 200
    compressed_ids = []
    for index in range(compressed_n):
        compressed_index = min(n - 1, start + int(float(n - 1 - start) / float(compressed_n) * float(index)))
        print(compressed_index)
        compressed_result.append(sorted_sorted_results[compressed_index])
        compressed_ids.append(sorted_sorted_results[compressed_index][-1])

    compressed_database = {}
    for id in compressed_ids:
        compressed_database[id] = database[id]

    with open(os.path.join(".", args.data_dir, "results_compressed.file"), 'wb') as f:
        pickle.dump(compressed_result, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(".", args.data_dir, "database_compressed.file"), 'wb') as f:
        pickle.dump(compressed_database, f, pickle.HIGHEST_PROTOCOL)
    '''

def inspect(args):
    from tensorflow.python.tools import inspect_checkpoint as chkp

    # print all tensors in checkpoint file
    chkp.print_tensors_in_checkpoint_file(os.path.join(args.logdir, "model.ckpt-1947700"), tensor_name='', all_tensors=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['training_direct', 'run_on_real_data','inspect',
                                           'train_on_real_data','train_on_real_data_fast',
                                           'train_on_all_data','finetune_test_data',
                                           'finetune_test_data_from_prior','finetune_test_data_from_prior_with_adam',
                                           'finetune_test_data_with_adam','plot_to_scale','compress_data','plot_param_histo'])
    parser.add_argument('--logdir', required=True)

    parser.add_argument('--batch_size', type=int, default=16+4)
    parser.add_argument('--virtual_batches', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-3/4.)


    parser.add_argument('--prob_choose_real', type=float, default=0.9)

    parser.add_argument('--number_of_prior_zarcs', type=int, default=9)
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--conv_filters', type=int, default=1*16)

    parser.add_argument('--num_conv', type=int, default=2)


    parser.add_argument('--percent_training', type=int, default=1)

    parser.add_argument('--total_steps', type=int, default=1000000)
    parser.add_argument('--checkpoint_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--nll_coeff', type=float, default=.1)
    parser.add_argument('--ordering_coeff', type=float, default=.5)
    parser.add_argument('--simplicity_coeff', type=float, default=.1)
    parser.add_argument('--sensible_phi_coeff', type=float, default=1.)
    parser.add_argument('--adam_beta1', type=float, default=.9)
    parser.add_argument('--adam_beta2', type=float, default=.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    parser.add_argument('--global_norm_clip', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=13311772)

    parser.add_argument('--data_dir', default='RealData')
    parser.add_argument('--plot_step', type=int, default=1000)
    parser.add_argument('--compressed_num', type=int, default=10000)

    parser.add_argument('--file_types', choices=['eis', 'fra'], default='fra')
    parser.add_argument('--histogram_file', default='results_of_inverse_model.file')

    parser.add_argument('--new_logdir', dest='new_logdir', action='store_true')
    parser.add_argument('--no-new_logdir', dest='new_logdir', action='store_false')
    parser.set_defaults(new_logdir=False)

    parser.add_argument('--visuals', dest='visuals', action='store_true')
    parser.add_argument('--no-visuals', dest='visuals', action='store_false')
    parser.set_defaults(visuals=False)

    parser.add_argument('--list_variables', dest='list_variables', action='store_true')
    parser.add_argument('--no-list_variables', dest='list_variables', action='store_false')
    parser.set_defaults(list_variables=False)

    parser.add_argument('--use_compressed', dest='use_compressed', action='store_true')
    parser.add_argument('--no-use_compressed', dest='use_compressed', action='store_false')
    parser.set_defaults(use_compressed=False)

    parser.add_argument('--test_fake_data', dest='test_fake_data', action='store_true')
    parser.add_argument('--no-test_fake_data', dest='test_fake_data', action='store_false')
    parser.set_defaults(test_fake_data=False)

    parser.add_argument('--dummy_frequencies', type=int, default=0)

    args = parser.parse_args()
    if args.mode == 'training_direct':
        training_direct(args)
    elif args.mode == 'run_on_real_data':
        run_on_real_data(args)

    elif args.mode == 'inspect':
        inspect(args)

    elif args.mode == 'train_on_real_data':
        train_on_real_data(args)

    elif args.mode == 'train_on_real_data_fast':
        train_on_real_data_fast(args)

    elif args.mode == 'train_on_all_data':
        train_on_all_data(args)
    elif args.mode == 'finetune_test_data':
        finetune_test_data(args)

    elif args.mode == 'finetune_test_data_from_prior':
        finetune_test_data_from_prior(args)

    elif args.mode == 'finetune_test_data_from_prior_with_adam':
        finetune_test_data_from_prior_with_adam(args)

    elif args.mode == 'finetune_test_data_with_adam':
        finetune_test_data_with_adam(args)

    elif args.mode == 'plot_to_scale':
        plot_to_scale(args)

    elif args.mode == 'compress_data':
        compress_data(args)

    elif args.mode == 'plot_param_histo':
        plot_param_histo(args)