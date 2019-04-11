#          NetBuilder
The neural network class is in NeuralNet.py.
It allows to easily create fully connected feedforward networks of
any size allowed by available memory. It uses numpy arrays as the primary
data structure for the weight matrices.
With this package you can create deep neural networks very quickly.

## Installation
Package can be installed with pip:

```sh
pip install netbuilder
```
## How to Use
The project's documentation was built using Sphinx and stored at readthedocs.org but for some reason it stopped working there. I need to rebuild it and that will happen when I find the time. In the meantime, the same documentation is spread throughout the source code as docstrings. I will include a small piece below:

To use the package, it has to be imported first like:
```
  >>> import netbuilder
  or
  >>> import netbuilder as nb

  With the following lines, you can create a neural network for
  a binary gate:

  >>> net = nb.Network()
  >>> net.init(topology=[2,1])

  The  first line above will create a `Network` object. The parameters
  of the network are not defined yet. The second line tells it to initialize
  weights for a shape of two input nodes for the first layer and one output
  node at the final layer.
  To create hidden layers, just add then to the topology parameter when
  initializing the network:

  >>> net = nb.Network()
  >>> net.init(topology=[2,5,5,1])

  The above lines will create a `Network` object with 4 layers: one input layer
  with 2 nodes, two hidden layers with 5 nodes each, and an output layer with
  one node.
  To perform a feedforward propagation an input array is needed. If the array
  is a numpy array with shape [number of samples x number of features], then
  the `feedfoward` method can be used:

  >>> x = numpy.array([[0,1]])
  >>> net.feedforward(x)
  array([[ 0.82683518]])

  Note above that the following format for x will cause an error because the
  shape is (,2) when it should be (1,2):

  >>> x = numpy.array([0,1])

  The method `predict` is available for quick testing without worry about the
  format of the input array:

  >>> x = [0,1]
  >>> net.predict(x)
  array([[ 0.82683518]])
```

And documentation built with Sphinx can be found at (not working for now):
http://netbuilder.readthedocs.io/en/latest/
