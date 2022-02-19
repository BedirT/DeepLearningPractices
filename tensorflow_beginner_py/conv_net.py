"""
Implementing a Convolutional Neural Network (CNN)
using TensorsFlow 2.0 and testing it on CIFAR-10.
"""
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets.cifar10 as cifar10

"""
Dataset related notes:
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 32x32 images
- RGB color channels
"""

class ConvNet:
    """
    Class ConvNet that defines a convolutional neural network
    with n hidden layers.
    """
    def __init__(self,
                 input_shape,
                 output_size,
                 optimizer='adam',
                 batch_size=32,
                 hidden_conv_layers=[(32, 3, 'valid', 2)],
                 hidden_dense_layers=[32]):
        """
        Initial Setup for the model.
        Args:
            input_shape: The shape of the input layer.
            output_size: The size of the output layer.
            optimizer: The optimizer to use. Default is Adam. 
                Available options are ['adam', 'sgd', 'rmsprop', 'adagrad']
            batch_size: The batch size to use.
            hidden_conv_layers: The convolutional layers to use. 
                The format is [(output_size, kernel_size, padding, pool_size)]
            hidden_dense_layers: The dense layers to use.
                The format is [output_size]
        """
        self._input_shape = input_shape
        self._output_size = output_size
        self._hidden_conv_layers = hidden_conv_layers
        self._hidden_dense_layers = hidden_dense_layers
        self._batch_size = batch_size

        # Setting up the model using the Sequential API
        self.build_model_sequential()

        # Setup the optimizer
        if optimizer == 'adam':
            self._optimizer = keras.optimizers.Adam()
        elif optimizer == 'sgd':
            self._optimizer = keras.optimizers.SGD()
        elif optimizer == 'rmsprop':
            self._optimizer = keras.optimizers.RMSprop()
        elif optimizer == 'adagrad':
            self._optimizer = keras.optimizers.Adagrad()
        else:
            raise ValueError("Invalid optimizer. Available options are ['adam', 'sgd', 'rmsprop', 'adagrad']")

    def build_model_sequential(self):
        """
        Building the model for the network using the sequential
        API.
        """
        self.layers = [keras.Input(shape=self._input_shape)] # 32, 32, 3

        # For Conv2D:
        # - Param1: How many channels to output
        # - Param2: Kernel size / if single int repeates accross dimensions
        # - Padding: 'same'/'valid' (default to valid)
        #   - 'same': The previous shape won't be effected; will stay the same.
        #   - 'valid':The previous shape will change according to the kernel size.
        for conv_layer in self._hidden_conv_layers:
            # 0: output size, 1: kernel size, 2: padding, 3: pooling size
            self.layers.append(layers.Conv2D(filters=conv_layer[0], 
                                             kernel_size=conv_layer[1],
                                             padding=conv_layer[2],
                                             activation='relu'))
            self.layers.append(layers.MaxPool2D(pool_size=conv_layer[3]))

        # Flatten the output of the last conv2D
        self.layers.append(layers.Flatten())

        # Add the dense layers
        for dense_layer in self._hidden_dense_layers:
            self.layers.append(layers.Dense(dense_layer, activation='relu'))
        
        # Output layer with output size
        self.layers.append(layers.Dense(self._output_size))

        self.model = keras.Sequential(self.layers)

        print(self.model.summary())

    def build_model_functional(self):
        """
        Building the model for the network using the functional
        API.
        """
        raise NotImplementedError("Functional API not implemented yet")

    def train(self, x_train, y_train, epochs=5, learning_rate=0.001):
        """Trains the model using x_train and y_train"""
        self._optimizer.learning_rate = learning_rate

        # Setup the training logic
        self.model.compile(
            # from logits true again since we dont use softmax for output layer
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=self._optimizer,
            metrics=["accuracy"]
        )

        # train using fit
        self.model.fit(x_train, y_train, batch_size=self._batch_size, epochs=epochs)
    
    def evaluate(self, x_test, y_test):
        """Evaluate how good a model is"""
        self.model.evaluate(x_test, y_test, batch_size=self._batch_size)

def test(unused_args):
    """Running the Network, testing"""
    # reading the data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # We are not reshaping as this is convnet and we make use of
    # the dimensionality.
    x_train = x_train.astype(dtype="float32") / 255.
    x_test = x_test.astype(dtype="float32") / 255.

    network = ConvNet((32, 32, 3), 10, 
                      hidden_conv_layers=[
                          (256, 3, 'valid', 2),
                          (512, 3, 'valid', 1),
                          (128, 3, 'valid', 2),
                      ],
                      hidden_dense_layers=[512, 256, 128])
    
    network.train(x_train, y_train, epochs=10)
    network.evaluate(x_test, y_test)

if __name__ == "__main__":
    test(None)