import sys
sys.path.append('../autograd')

from value import Value #type: ignore
import random

random.seed(42)

w = Value(random.uniform(-1, 1))
b = Value(random.uniform(-1, 1))

training_data = [
    (Value(0.5), Value(1.0)),
    (Value(1.0), Value(2.0)),
    (Value(1.5), Value(3.0)),
    (Value(2.0), Value(4.0)),
]

learning_rate = 0.05

for step in range(1001):
    # 1. Forward pass — accumulate loss across all training points
    total_loss = Value(0)
    for x, y_target in training_data:
        y_pred = w * x + b
        total_loss = total_loss + (y_pred - y_target) ** 2
    mse = total_loss * Value(1 / len(training_data))
    
    # 2. Zero gradients
    w.grad = 0
    b.grad = 0
    
    # 3. Backward pass
    mse.backward()
    
    # 4. Update weights
    w.data -= learning_rate * w.grad
    b.data -= learning_rate * b.grad
    
    # 5. Print every 10 steps so output isn't overwhelming
    if step % 100 == 0:
        print(f"step {step}: mse = {mse.data:.6f}, w = {w.data:.4f}, b = {b.data:.4f}")


print("\nGeneralization test (inputs not in training data):")
for x_test in [0.3, 0.7, 1.2, 2.5]:
    x_val = Value(x_test)
    y_pred = w * x_val + b
    y_true = 2 * x_test
    print(f"x = {x_test}: predicted {y_pred.data:.4f}, true {y_true:.4f}")