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



a = Value(3)
b = Value(4)
c = Value(2)
d = a * b + c
print(d.data)       
print(d._prev)      
print(d._op)        