from keras.layers import Layer, InputSpec
from keras import backend as K


class Repeat3DVector(Layer):
    """Repeats the input n times.

    # Arguments
        n: integer, repetition factor.

    # Input shape
        3D tensor of shape `(B, T, F)`.

    # Output shape
        4D tensor of shape `(B, T, n, F)`.
    """

    def __init__(self, n, **kwargs):
        super(Repeat3DVector, self).__init__(**kwargs)
        self.n = n
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n, input_shape[2])

    def call(self, inputs):
        x = K.expand_dims(inputs, axis=2)
        return K.tile(x, [1, 1, self.n, 1])

    def get_config(self):
        config = {'n': self.n}
        base_config = super(Repeat3DVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
