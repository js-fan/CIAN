from .layers import *

def wResStem(data, num_filter, momentum=0.9, eps=1e-5, use_global_stats=False, bn_data=True,
        name=None, lr_mult=1, reuse=None):
    name = '' if name is None else name
    if bn_data:
        x = BN(data, fix_gamma=True, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
               name=name+'bn_data', reuse=reuse)
    else:
        x = data

    x = Conv(x, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True,
            name=name+'conv1a', lr_mult=lr_mult, reuse=reuse)
    return x


def wResUnit(data, num_filter, stride, dilate, projection, bottle_neck, dropout=0, momentum=0.9, eps=1e-5,
        use_global_stats=False, name=None, lr_mult=1, reuse=None, **kwargs):
    assert name is not None

    x = BNRelu(data, fix_gamma=False, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
               name='bn'+name+'_branch2a', lr_mult=lr_mult, reuse=reuse)

    if projection:
        shortcut = Conv(x, num_filter=num_filter, kernel=(1, 1), stride=(stride,)*2,
                pad=(0, 0), no_bias=True, name='res'+name+'_branch1', lr_mult=lr_mult, reuse=reuse)
    else:
        shortcut = data

    if bottle_neck:
        x = Conv(x, num_filter=int(num_filter/4.), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                no_bias=True, name='res'+name+'_branch2a', lr_mult=lr_mult, reuse=reuse)
        x = BNRelu(x, fix_gamma=False, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
                name='bn'+name+'_branch2b1', lr_mult=lr_mult, reuse=reuse)
        if dropout > 0:
            x = Drop(x, p=dropout)
        x = Conv(x, num_filter=int(num_filter/2.), kernel=(3, 3), stride=(stride,)*2, pad=(dilate,)*2,
                dilate=(dilate,)*2, no_bias=True, name='res'+name+'_branch2b1', lr_mult=lr_mult, reuse=reuse)
        x = BNRelu(x, fix_gamma=False, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
                name='bn'+name+'_branch2b2', lr_mult=lr_mult, reuse=reuse)
        if dropout > 0:
            x = Drop(x, p=dropout)
        x = Conv(x, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                no_bias=True, name='res'+name+'_branch2b2', lr_mult=lr_mult, reuse=reuse)
    else:
        #mid_filter = num_filter//2 if name in ['5', '5a', '5b1', '5b2'] else num_filter
        mid_filter = kwargs.get('mid_filter', num_filter)
        fst_dilate = kwargs.get('fst_dilate', dilate)
        x = Conv(x, num_filter=mid_filter, kernel=(3, 3), stride=(stride,)*2, pad=(fst_dilate,)*2,
                dilate=(fst_dilate,)*2, no_bias=True, name='res'+name+'_branch2a', lr_mult=lr_mult, reuse=reuse)
        x = BNRelu(x, fix_gamma=False, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
                name='bn'+name+'_branch2b1', lr_mult=lr_mult, reuse=reuse)
        x = Conv(x, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(dilate,)*2,
                dilate=(dilate,)*2, no_bias=True, name='res'+name+'_branch2b1', lr_mult=lr_mult, reuse=reuse)

    x = x + shortcut    
    return x


def wResBlock(data, num_unit, num_filter, stride, dilate, bottle_neck, dropout=0, momentum=0.9, eps=1e-5,
        use_global_stats=False, name=None, lr_mult=1, reuse=None, **kwargs):
    assert name is not None
    x = wResUnit(data, num_filter, stride, dilate, True, bottle_neck, dropout, momentum, eps,
            use_global_stats, name=name+'a', lr_mult=lr_mult, reuse=reuse, **kwargs)
    for i in range(1, num_unit):
        x = wResUnit(x, num_filter, 1, dilate, False, bottle_neck, dropout, momentum, eps,
                use_global_stats, name=name+'b%d'%i, lr_mult=lr_mult, reuse=reuse, **kwargs)
    return x


def wresnet38(x, momentum=0.9, eps=1e-5, use_global_stats=False, name=None, out_internals=False, lr_mult=1, reuse=None):
    name = '' if name is None else name
    internals = []

    x = wResStem(x, 64, momentum, eps, use_global_stats, bn_data=True, name=name, lr_mult=lr_mult, reuse=reuse)
    x = wResBlock(x, 3, 128, 2, 1, False, 0, momentum, eps, use_global_stats, name+'2', lr_mult, reuse)
    x = wResBlock(x, 3, 256, 2, 1, False, 0, momentum, eps, use_global_stats, name+'3', lr_mult, reuse)
    x = wResBlock(x, 6, 512, 2, 1, False, 0, momentum, eps, use_global_stats, name+'4', lr_mult, reuse)
    x = wResBlock(x, 3, 1024, 1, 2, False, 0, momentum, eps, use_global_stats, name+'5', lr_mult, reuse, mid_filter=512, fst_dilate=1)
    internals.append(x)
    x = wResBlock(x, 1, 2048, 1, 4, True, 0.3, momentum, eps, use_global_stats, name+'6', lr_mult, reuse)
    internals.append(x)
    x = wResBlock(x, 1, 4096, 1, 4, True, 0.5, momentum, eps, use_global_stats, name+'7', lr_mult, reuse)
    internals.append(x)
    x = BNRelu(x, fix_gamma=False, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
            name=name+'bn7', lr_mult=lr_mult, reuse=reuse)

    if out_internals:
        return x, internals
    else:
        return x



