from .layers import *

def incepConv(data, num_filter, kernel, stride=None, dilate=None, pad=None, momentum=0.9, eps=1e-5,
        use_global_stats=False, name=None, lr_mult=1, reuse=None):
    assert name is not None
    x = Conv(data, num_filter, kernel, stride, dilate, pad, name='conv_%s'%name, lr_mult=lr_mult, reuse=reuse)
    x = BN(x, fix_gamma=False, momentum=momentum, eps=eps, use_global_stats=use_global_stats, 
            name='bn_%s'%name, lr_mult=lr_mult, reuse=reuse)
    x = Relu(x)
    return x

def incepBlockA(data, num_filter_1, num_filter_3r, num_filter_3, num_filter_d3r, num_filter_d3, num_filter_p,
        pool_type, dilate=1, momentum=0.9, eps=1e-5, use_global_stats=False,
        name=None, lr_mult=1, reuse=None):
    assert name is not None

    x1 = incepConv(data, num_filter_1, (1, 1), momentum=momentum, eps=eps, use_global_stats=use_global_stats,
            name='%s_1x1'%name, lr_mult=lr_mult, reuse=reuse)

    x3 = incepConv(data, num_filter_3r, (1, 1), momentum=momentum, eps=eps, use_global_stats=use_global_stats,
            name='%s_3x3_reduce'%name, lr_mult=lr_mult, reuse=reuse)
    x3 = incepConv(x3, num_filter_3, (3, 3), pad=(dilate,)*2, dilate=(dilate,)*2, momentum=momentum, eps=eps,
            use_global_stats=use_global_stats, name='%s_3x3'%name, lr_mult=lr_mult, reuse=reuse)

    xd3 = incepConv(data, num_filter_d3r, (1, 1), momentum=momentum, eps=eps, use_global_stats=use_global_stats,
            name='%s_double_3x3_reduce'%name, lr_mult=lr_mult, reuse=reuse)
    xd3 = incepConv(xd3, num_filter_d3, (3, 3), pad=(dilate,)*2, dilate=(dilate,)*2, momentum=momentum, eps=eps,
            use_global_stats=use_global_stats, name='%s_double_3x3_0'%name, lr_mult=lr_mult, reuse=reuse)
    xd3 = incepConv(xd3, num_filter_d3, (3, 3), pad=(dilate,)*2, dilate=(dilate,)*2, momentum=momentum, eps=eps,
            use_global_stats=use_global_stats, name='%s_double_3x3_1'%name, lr_mult=lr_mult, reuse=reuse)

    xp = Pool(data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool_type)
    xp = incepConv(xp, num_filter_p, (1, 1), momentum=momentum, eps=eps, use_global_stats=use_global_stats,
            name='%s_proj'%name, lr_mult=lr_mult, reuse=reuse)

    x = mx.sym.Concat(x1, x3, xd3, xp, dim=1, name='ch_concat_%s_chconcat'%name)
    return x

def incepBlockB(data, num_filter_3r, num_filter_3, num_filter_d3r, num_filter_d3,
        stride=2, dilate=1, momentum=0.9, eps=1e-5, use_global_stats=False,
        name=None, lr_mult=1, reuse=None):
    assert name is not None

    x3 = incepConv(data, num_filter_3r, (1, 1), momentum=momentum, eps=eps, use_global_stats=use_global_stats,
            name='%s_3x3_reduce'%name, lr_mult=lr_mult, reuse=reuse)
    x3 = incepConv(x3, num_filter_3, (3, 3), stride=(stride,)*2, pad=(dilate,)*2, dilate=(dilate,)*2,
            momentum=momentum, eps=eps, use_global_stats=use_global_stats, name='%s_3x3'%name, lr_mult=lr_mult, reuse=reuse)
    
    xd3 = incepConv(data, num_filter_d3r, (1, 1), momentum=momentum, eps=eps, use_global_stats=use_global_stats,
            name='%s_double_3x3_reduce'%name, lr_mult=lr_mult, reuse=reuse)
    xd3 = incepConv(xd3, num_filter_d3, (3, 3), stride=(1, 1), pad=(dilate,)*2, dilate=(dilate,)*2,
            momentum=momentum, eps=eps, use_global_stats=use_global_stats,
            name='%s_double_3x3_0'%name, lr_mult=lr_mult, reuse=reuse)
    xd3 = incepConv(xd3, num_filter_d3, (3, 3), stride=(stride,)*2, pad=(dilate,)*2, dilate=(dilate,)*2,
            momentum=momentum, eps=eps, use_global_stats=use_global_stats,
            name='%s_double_3x3_1'%name, lr_mult=lr_mult, reuse=reuse)
    
    xp = Pool(data, kernel=(3, 3), stride=(stride,)*2, pad=(1, 1), pool_type='max')

    x = mx.sym.Concat(x3, xd3, xp, dim=1, name='ch_concat_%s_chconcat'%name)
    return x


def inceptionBN(x, momentum=0.9, eps=1e-5, use_global_stats=False, bn_data=True, name=None, lr_mult=1, reuse=None):
    name = '' if name is None else name
    if bn_data:
        x = BN(x, fix_gamma=True, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
               name=name+'bn_data', reuse=reuse)

    x = incepConv(x, 64, (7, 7), stride=(2, 2), pad=(3, 3), name=name+'1', lr_mult=lr_mult, reuse=reuse)
    x = Pool(x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), name=name+'pool_1', pool_type='max')

    x = incepConv(x, 64,  (1, 1), stride=(1, 1), pad=(0, 0), name=name+'2_red', lr_mult=lr_mult, reuse=reuse)
    x = incepConv(x, 192, (3, 3), stride=(1, 1), pad=(1, 1), name=name+'2', lr_mult=lr_mult, reuse=reuse)
    x = Pool(x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), name=name+'pool_2', pool_type='max')

    x = incepBlockA(x, 64, 64, 64, 64, 96, 32, 'avg', 1, momentum, eps, use_global_stats, '3a', lr_mult, reuse)
    x = incepBlockA(x, 64, 64, 96, 64, 96, 64, 'avg', 1, momentum, eps, use_global_stats, '3b', lr_mult, reuse)
    x = incepBlockB(x, 128, 160, 64, 96, 1, 2, momentum, eps, use_global_stats, '3c', lr_mult, reuse)

    x = incepBlockA(x, 224, 64, 96, 96, 128, 128, 'avg', 2, momentum, eps, use_global_stats, '4a', lr_mult, reuse)
    x = incepBlockA(x, 192, 96, 128, 96, 128, 128, 'avg', 2, momentum, eps, use_global_stats, '4b', lr_mult, reuse)
    x = incepBlockA(x, 160, 128, 160, 128, 160, 128, 'avg', 2, momentum, eps, use_global_stats, '4c', lr_mult, reuse)
    x = incepBlockA(x, 96, 128, 192, 160, 192, 128, 'avg', 2, momentum, eps, use_global_stats, '4d', lr_mult, reuse)
    x = incepBlockB(x, 128, 192, 192, 256, 1, 4, momentum, eps, use_global_stats, '4e', lr_mult, reuse)

    x = incepBlockA(x, 352, 192, 320, 160, 224, 128, 'avg', 4, momentum, eps, use_global_stats, '5a', lr_mult, reuse)
    x = incepBlockA(x, 352, 192, 320, 192, 224, 128, 'max', 4, momentum, eps, use_global_stats, '5b', lr_mult, reuse)
    return x


