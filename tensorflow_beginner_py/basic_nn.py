"""
Implementing a basic linear neural network commenting
every step of the process, and a simple test using MNIST.
"""
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets.mnist as mnist


class NeuralNetwork:
    """
    Class NeuralNetwork that defines a neural network with one hidden
    layer performing binary classification.
    """

    def __init__(self, 
                 input_size,
                 output_size,
                 optimizer='adam',
                 batch_size=32,
                 hidden_layers=[32]):
        """
        Class constructor.
        Args:
            input_size: The size of the input layer.
            output_size: The size of the output layer.
            optimizer: The optimizer to use. Default is adam. Available optimizers:
                - [adam, sgd, rmsprop, adagrad]
            batch_size: The batch size to use.
        """
        self._batch_size = batch_size
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_layers = hidden_layers
        # Set up the layers with given sizes
        self.build_model_functional()
        # self.build_model_sequential()

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
            raise ValueError(f"Optimizer {optimizer} not supported.")
        
    def build_model_sequential(self):
        """
        Using the Sequential API (Less flexible API - single i/o) to build
        the network model.
        """
        # input layer
        self.layers = [keras.Input(shape=self._input_size)]

        for layer_size in self._hidden_layers:
            # Dense is a fully connected layer
            self.layers.append(layers.Dense(layer_size, activation='relu'))

        # Setup the output layer - no activation
        self.layers.append(layers.Dense(self._output_size))

        # Set the layers in keras seq.
        self.model = keras.Sequential(self.layers)

        print(self.model.summary())

        # ALTERNATIVE IMPLEMENTATION:
        # We could also use keras.Sequential().add() instead of building from
        # a list. This is in particular useful to debug (print model summary
        # after all.)

    def build_model_functional(self):
        """
        Building the model using the Functional API (A bit more flexible).
        (Doing the same exact things as in sequential api)
        """
        inputs = keras.Input(shape=self._input_size)
        x = layers.Dense(self._hidden_layers[0], activation='relu', name='Hidden_Layer_1')(inputs)
        for idx in range(1, len(self._hidden_layers)):
            x = layers.Dense(self._hidden_layers[idx], activation='relu', name=f'Hidden_Layer_{idx+1}')(x)
        outputs = layers.Dense(self._output_size)(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def extract_layer_output(self, x_train, layer_idx=None, layer_str=""):
        """
        Showing how to access the model features on a specific layer.
        """
        # layer id and str cant be set at the same time, and one should be set
        assert not (layer_idx and layer_str)
        assert layer_idx or layer_str

        if layer_str:
            # we are accessing the layer_str named layer
            outputs = [self.model.get_layer(layer_str).output]
        if layer_idx:
            # we are accessing the layer_idx th layer
            outputs = [self.model.layers[layer_idx].output]

        # overwrite the model to access the specific layer
        test_model = keras.Model(inputs=self.model.inputs,
                                 outputs=outputs)
        
        return test_model.predict(x_train)

    def train(self,
              x_train,
              y_train,
              num_epochs=10,
              learning_rate=0.001):
        """
        Train the model.
        Args:
            x_train: The training input data.
            y_train: The training labels.
            num_epochs: The number of epochs to train the model.
            learning_rate: The learning rate to use.
        """
        # Update the learning rate
        self._optimizer.learning_rate = learning_rate

        # Configuring the training logic of the network
        self.model.compile(
            # from_logits, to add softmax to the outputs
            # Sparse is used if y label is exact value rather than one-hot vc.
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=self._optimizer,
            metrics=["accuracy"]
        )

        # Specify the train details and train
        self.model.fit(x_train, y_train, batch_size=self._batch_size, epochs=num_epochs)

    def evaluate(self, x_test, y_test):
        # Evaluate how good the mode did
        self.model.evaluate(x_test, y_test, batch_size=self._batch_size)  


def test(unused_args):
    """
    Function test that tests the neural network implementation
    using MNIST.
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data
    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.

    # setting up the network according to the data.
    network = NeuralNetwork(input_size=(28*28), output_size=10,
                            optimizer='adam', batch_size=32, 
                            hidden_layers=[128, 256, 128])
    
    # Training and then evaluating the data using the functions we wrote.
    network.train(x_train, y_train)
    network.evaluate(x_test, y_test)

    # Getting the features from a specific layer could be useful
    # when debugging.
    feature = network.extract_layer_output(x_train, layer_idx=1)
    # print(feature)

if __name__ == '__main__':
    test(None)