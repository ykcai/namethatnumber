# namethatnumber

Handwritten Digit Recognition on the MNIST Dataset in Python using the Keras Deep Learning Library.

# Input Number of features

```python
features = input("Number of features for each layer: ")
```

# Input the Optimizer Learning Rate

```python
learningRate = input("Learning rate of optimizer: ")
```

# Change the number of layers 1 to 3

```python
def baseline_model():
	# create model
	...
		#remove this layer to get 2 layers
	model.add(Conv2D(int(features), (3, 3), activation='relu'))
	model.add(Flatten())
		#remove this layer to get 1 layer
	model.add(Dense(int(features), activation='relu'))
  ...
```

# Run

`python SimpleCNN.py` and enter the parameters.
