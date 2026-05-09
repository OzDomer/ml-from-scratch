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

        


a = Value(3)
b = Value(4)
e = a * b           
f = a + b           
d = e + f           
d.backward()
d.print_graph()
print(a.grad)       
print(b.grad)       
print(d.data)       