import tensorflow as tf
from keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects



# Custom Keras layer that applies a Fourier transformation using learned sine and cosine kernels.
class FourierLayer(Layer):
    """
    Custom Keras layer that applies a Fourier transformation to the input tensor
    using learned sine and cosine kernels.

    Attributes:
    - output_dim (int): Number of output dimensions for the Fourier transformation.
    """

    def __init__(self, output_dim: int, **kwargs):
        """
        Initializes the FourierLayer with the specified output dimensions.

        Parameters:
        - output_dim (int): Number of output dimensions for the Fourier transformation.
        - kwargs: Additional keyword arguments for the Layer class.
        """
        self.output_dim = output_dim
        super(FourierLayer, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        """
        Builds the layer by initializing the sine and cosine kernels used
        for the Fourier transformation.

        Parameters:
        - input_shape (tf.TensorShape): Shape of the input tensor.
        """
        self.kernel_sin = self.add_weight(name='kernel_sin',
                                          shape=(self.output_dim,),
                                          initializer='glorot_uniform',
                                          trainable=True)
        self.kernel_cos = self.add_weight(name='kernel_cos',
                                          shape=(self.output_dim,),
                                          initializer='glorot_uniform',
                                          trainable=True)
        super(FourierLayer, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies the Fourier transformation to the input tensor using
        the learned sine and cosine kernels.

        Parameters:
        - x (tf.Tensor): Input tensor.

        Returns:
        - tf.Tensor: Transformed tensor.
        """
        result = tf.sin(tf.multiply(x, self.kernel_sin)) + tf.cos(tf.multiply(x, self.kernel_cos))
        return result

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """
        Computes the output shape of the layer, which matches the input shape.

        Parameters:
        - input_shape (tf.TensorShape): Shape of the input tensor.

        Returns:
        - tf.TensorShape: Shape of the output tensor.
        """
        return input_shape

get_custom_objects().update({'FourierLayer': FourierLayer})

class Activations:
    @staticmethod
    def custom_activation(x: tf.Tensor) -> tf.Tensor:
        """
        Custom activation function combining linear and non-linear transformations.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Transformed tensor.
        """
        return x + K.square(K.sin(x))


    @staticmethod
    def sinusoidal_activation(x: tf.Tensor) -> tf.Tensor:
        """
        Sinusoidal activation function that applies a square sine transformation to the input.

        Parameters:
        - x (tf.Tensor): Input tensor.

        Returns:
        - tf.Tensor: Transformed tensor.
        """
        return K.square(K.sin(x))
    
    get_custom_objects().update({'custom_activation': custom_activation})
    get_custom_objects().update({'sinusoidal_activation': sinusoidal_activation})