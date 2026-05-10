w = Value(random.uniform(-1, 1))
b = Value(random.uniform(-1, 1))
x = Value(0.5)
y_target = Value(1.0)

learning_rate = 0.1

learning_rate = 0.1

for step in range(50):
    # 1. Forward pass — rebuild graph
    y_pred = w * x + b
    loss = (y_pred - y_target) ** 2
    
    # 2. Zero gradients
    w.grad = 0
    b.grad = 0
    
    # 3. Backward pass
    loss.backward()
    
    # 4. Update weights
    w.data -= learning_rate * w.grad
    b.data -= learning_rate * b.grad
    
    print(f"step {step}: loss = {loss.data:.6f}, y_pred = {y_pred.data:.6f}")

