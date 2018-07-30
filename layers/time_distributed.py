from keras import backend as K
from keras.layers import wrappers
from keras.utils.generic_utils import has_arg
from keras.engine.topology import _object_list_uid


class TimeDistributed(wrappers.TimeDistributed):
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return super().compute_output_shape(input_shape)

    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        super().build(input_shape)
        self.input_spec = None

    def call(self, inputs, training=None, mask=None):
        kwargs = {}
        if has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        uses_learning_phase = False

        if not isinstance(inputs, list):
            inputs = [inputs]

        reshaped_inputs = []
        for input in inputs:
            input_shape = K.int_shape(input)

            # No batch size specified, therefore the layer will be able
            # to process batches of any size.
            # We can go with reshape-based implementation for performance.
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(input)[1]
            # Shape: (num_samples * timesteps, ...). And track the
            # transformation in self._input_map.
            input_uid = _object_list_uid(input)
            input = K.reshape(input, (-1,) + input_shape[2:])
            self._input_map[input_uid] = input

            reshaped_inputs.append(input)

        # (num_samples * timesteps, ...)
        reshaped_initial_states = reshaped_inputs[1:]
        reshaped_inputs = reshaped_inputs[0]
        if reshaped_initial_states:
            kwargs['initial_state'] =  reshaped_initial_states

        y = self.layer.call(reshaped_inputs, **kwargs)

        if hasattr(y, '_uses_learning_phase'):
            uses_learning_phase = y._uses_learning_phase
        # Shape: (num_samples, timesteps, ...)
        input_shape = [K.int_shape(input) for input in inputs]
        output_shape = self.compute_output_shape(input_shape)
        y = K.reshape(y, (-1, input_length) + output_shape[2:])

        # Apply activity regularizer if any:
        if (hasattr(self.layer, 'activity_regularizer') and
           self.layer.activity_regularizer is not None):
            regularization_loss = self.layer.activity_regularizer(y)
            self.add_loss(regularization_loss, inputs)

        if uses_learning_phase:
            y._uses_learning_phase = True
        return y
