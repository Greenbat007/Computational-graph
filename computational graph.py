import math


class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

        
    def __repr__(self):
        return f"Value({self.label}={self.data})"
    

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
   
 
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

  
    def __neg__(self): # -self
        return self * -1


    def __radd__(self, other): # other + self
        return self + other


    def __sub__(self, other): # self - other
        return self + (-other)


    def __rsub__(self, other): # other - self
        return other + (-self)


    def __rmul__(self, other): # other * self
        return self * other
    
    #finds grad with all nodes behind the node called.
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        print(topo)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

#Forward Pass
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(7, label='b')
x1w1 = x1 * w1; x1w1.label='x1*w1'
x2w2 = x2 * w2; x2w2.label='x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'

#Backward Pass
n.backward()
print(x1.grad)

