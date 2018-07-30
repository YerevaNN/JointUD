from keras import backend as K
from keras.layers import GRUCell
from keras.activations import softmax


class RecurrentCell(GRUCell):
    def __init__(self, units, dense_units=None, **kwargs):
        super(RecurrentCell, self).__init__(units, **kwargs)
        self.dense_units = dense_units if dense_units is not None else 160
        self.state_size = (self.units, self.dense_units)

    def build(self, input_shape):
        super(RecurrentCell, self).build(input_shape)

        self.dense_kernel = self.add_weight(shape=(self.units, self.dense_units),
            initializer='glorot_uniform',
            name='dense_kernel',
            regularizer=None,
            constraint=None)

        self.dense_bias = self.add_weight(shape=self.dense_units,
            initializer='zeros',
            name='dense_bias',
            regularizer=None,
            constraint=None)

    def call(self, inputs, states, training=None):
        state_t_h = states[0]
        state_t_l = states[1]

        training_phase = training

        if training_phase:
            gru_input = inputs
        else:
            slice_index = -self.dense_units
            gru_input = K.concatenate([inputs[:, :slice_index], state_t_l])

        output_t, state_t_next_h = super(RecurrentCell, self).call(gru_input, state_t_h, training=training)

        output_t = K.dot(output_t, self.dense_kernel)
        output_t = K.bias_add(output_t, self.dense_bias)
        output_t = softmax(output_t, axis=-1)

        if training_phase:
            state_t_next_l = state_t_l
        else:
            state_t_next_l = output_t

        return output_t, state_t_next_h + [state_t_next_l]
