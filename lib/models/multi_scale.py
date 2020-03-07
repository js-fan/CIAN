from .layers import *

def MultiScale(scales):
    scales = [s for s in scales if s != 1]

    def func_wrapper(model_func):
        def model_func_ms(*args, **kwargs):
            assert len(args) > 0, 'Cannot find input variable'
            input_var = args[0]
            args = args[1:]

            out_0 = model_func(*((input_var,) + args), **kwargs)
            assert len(out_0) == 1, 'Only single output implemented'

            reuse = kwargs.get('reuse', None)
            if reuse is None:
                reuse = out_0
            if 'reuse' in kwargs:
                del kwargs['reuse']

            is_tensor4d = len(out_0.infer_shape(data=(1, 3, 100, 100))[1][0]) == 4

            out_ms = [out_0]
            for scale in scales:
                input_var_s = mx.sym.Custom(input_var, scale=scale, op_type='BilinearScale')
                out_s = model_func(*((input_var_s,) + args), reuse=reuse, **kwargs)
                if is_tensor4d:
                    out_s = mx.sym.Custom(out_s, out_0, op_type='BilinearScaleLike')
                out_ms.append(out_s)

            out_max = out_ms[0]
            for out_s in out_ms[1:]:
                out_max = mx.sym.maximum(out_max, out_s)
            out_ms.append(out_max)

            return mx.sym.Group(out_ms)
        return model_func_ms
    return func_wrapper
