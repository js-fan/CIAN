from ..layers import *

class DropConnect(mx.operator.CustomOp):
    def __init__(self, p):
        self.drop_rate = p
        self.mask = None
    
    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]
        if is_train or self.drop_rate == 0:
            mask_shape = [data.shape[0]] + [1] * (len(data.shape)-1)
            mask = mx.nd.random.uniform(0, 1, mask_shape, ctx=data.context)
            mask = (mask > self.drop_rate) / (1 - self.drop_rate)
            out = data * mask
            self.mask = mask
        else:
            out = data
            self.mask = None
        self.assign(out_data[0], req[0], out)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if self.mask is None:
            grad = out_grad[0].copy()
        else:
            grad = out_grad[0] * self.mask

        self.assign(in_grad[0], req[0], grad)

@mx.operator.register("DropConnect")
class DropConnectProp(mx.operator.CustomOpProp):
    def __init__(self, p):
        super(DropConnectProp, self).__init__(need_top_grad=True)
        self.drop_rate = float(p)
        assert self.drop_rate >= 0 and self.drop_rate < 1
    
    def list_arguments(self):
        return ['data']
    
    def list_outputs(self):
        return ['output']
    
    def infer_shape(self, in_shape):
        return in_shape, in_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return DropConnect(self.drop_rate)
