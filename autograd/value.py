class Value:
    def __init__(self, data, _children=(),_op=""):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Value(data={self.data})"

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * Value(-1)

    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting constant powers"
        out = Value(self.data ** other, (self,), f"**{other}")
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(max(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)
                topo.append(node)
        
        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()
    
    def print_graph(self, depth=0):
        indent = "  " * depth
        op_label = f" op={self._op}" if self._op else ""
        print(f"{indent}Value(data={self.data}){op_label}")
        for parent in self._prev:
            parent.print_graph(depth + 1)      

        

# Test the positive case
a = Value(3)
b = a.relu()
print(b.data)        # 3 (passes through unchanged)
b.backward()
print(a.grad)        # 1 (slope is 1 for positive inputs)

# Test the negative case
c = Value(-2)
d = c.relu()
print(d.data)        # 0 (zeroed out)
d.backward()
print(c.grad)        # 0 (slope is 0 for negative inputs)