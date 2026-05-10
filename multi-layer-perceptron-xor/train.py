import sys
sys.path.append('../autograd')

from value import Value  # type: ignore
import random

random.seed(42)

class Neuron:
    def __init__(self, num_inputs, use_relu=True):
        # Create num_inputs random weights, one random bias
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.bias = Value(random.uniform(-1, 1))
        self.use_relu = use_relu

    def __call__(self, inputs):
        act = self.bias
        for w, x in zip(self.weights, inputs):
            act += w * x
        return act.relu() if self.use_relu else act

    def parameters(self):    # ← NEW
        return self.weights + [self.bias]
     

class Layer:
    def __init__(self, num_inputs, num_neurons, use_relu=True):
        # Create num_neurons neurons, each with num_inputs inputs
        self.neurons = [Neuron(num_inputs, use_relu=use_relu) for _ in range(num_neurons)]
    
    def __call__(self, inputs):
        # Run every neuron on the same inputs, collect outputs into a list
        return [neuron(inputs) for neuron in self.neurons] 

    def parameters(self):    # ← NEW
       return [param for neuron in self.neurons for param in neuron.parameters()]



class MLP:
    def __init__(self, num_inputs, layer_sizes):
        # layer_sizes is a list, e.g., [3, 1] meaning hidden layer of 3, output layer of 1
        # Build a layer for each size, with input size matching previous layer's output
        sizes = [num_inputs] + layer_sizes
        self.layers = []
        for i in range(len(layer_sizes)):
            in_size = sizes[i]
            out_size = sizes[i + 1]
            is_last = (i == len(layer_sizes) - 1)
            self.layers.append(Layer(in_size, out_size, use_relu=not is_last))
    
    
    def __call__(self, inputs):
        # Run inputs through each layer in sequence
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    
    def parameters(self):    # ← NEW
        return [param for layer in self.layers for param in layer.parameters()]


xor_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

mlp = MLP(num_inputs=2, layer_sizes=[8, 1])

print(f"Total parameters: {len(mlp.parameters())}")

learning_rate = 0.1

for step in range(1000):
    # 1. Forward pass — accumulate total loss across all 4 points
    total_loss = Value(0)
    for inputs, target in xor_data:
        y_pred = mlp(inputs)[0]    # output layer returns a list; grab the single Value
        total_loss = total_loss + (y_pred - target) ** 2
    
    # 2. Zero gradients
    for p in mlp.parameters():
        p.grad = 0
    
    # 3. Backward pass
    total_loss.backward()
    
    # 4. Update parameters
    for p in mlp.parameters():
        p.data -= learning_rate * p.grad
    
    # 5. Print every 100 steps
    if step % 100 == 0:
        print(f"step {step}: loss = {total_loss.data:.6f}")


print("\nPredictions:")
for inputs, target in xor_data:
    y_pred = mlp(inputs)[0]
    print(f"input {inputs} → predicted {y_pred.data:.4f}, target {target}")