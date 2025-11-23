import os
import cupy as cp
import numpy as np
import math

##### QUICK ENABLE FOR TENSOR CORE OPS (for cupy only) ###
device = cp.cuda.Device()
# string containing the major index and the minor index. 
# For example, compute capability 3.5 is represented by the string ‘35’.
cc_major, cc_minor = device.compute_capability 
if int(cc_major) >= 8:
    os.environ["CUPY_TF32"] = "1"
##########################################


class Operation:
    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    
class GradTensor:
    def __init__(self, params):
        self.params = params
        self.shape = params.shape
        self.grade = None

    def zero_grad(self):
        self.grad = None

class GradLayer(Operation):

    def parameters(self):
        params = []
        for attr_name, attr_values in self.__dict__.items():
            if isinstance(attr_values, GradTensor):
                params.append(attr_values)
            
            elif isinstance(attr_values, GradLayer):
                params.extend(attr_values.parameters())

        return params


class LinearLayer(GradLayer):
    """
        y = x@w + b
        Dimensions: 
            x = [batches x features]
            w = [features x output]
            b = [1 x 1]
            y = [input x output]
    """
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        self.weight = GradTensor(
            cp.random.normal(
                scale=0.02,
                size=(in_features, out_features),
                dtype=cp.float32
            )
        )

        if bias:
            self.bias = GradTensor(
                cp.zeros((1, out_features), dtype=cp.float32)
            )
        else:
            self.bias = None
    
    def forward(self, x):
        self.x = x
        out = cp.matmul(x, self.weight.params)
        if self.bias:
            out += self.bias.params
        return out
    
    def backward(self,grad_output):
        self.weight.grad = np.matmul(self.x.T,grad_output)
        # Grad w.r.t bias
        if self.bias is not None:
            self.bias.grad = cp.sum(grad_output, axis=0, keepdims=True)
        
        # Grad w.r.t input
        return cp.matmul(grad_output, self.weight.params.T)
        
    

    

    