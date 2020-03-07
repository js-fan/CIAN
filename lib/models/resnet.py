from .layers import *

def ResStem(data, num_filter, momentum=0.9, eps=1e-5, use_global_stats=False, bn_data=True,
            name=None, lr_mult=1, reuse=None):
    name = '' if name is None else name
    if bn_data:
        x = BN(data, fix_gamma=True, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
               name=name+'bn_data', reuse=reuse)
    else:
        x = data

    x = Conv(x, num_filter=num_filter, kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True,
             name=name+'conv0', lr_mult=lr_mult, reuse=reuse)
    x = BN(x, fix_gamma=False, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
             name=name+'bn0', lr_mult=lr_mult, reuse=reuse)
    x = Relu(x, name=name+'relu0')
    x = Pool(x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name=name+'pool0')
    return x

def ResUnit(data, num_filter, stride, dilate, projection, bottle_neck, momentum=0.9, eps=1e-5,
            use_global_stats=False, name=None, lr_mult=1, reuse=None):
    assert name is not None
    x = BNRelu(data, fix_gamma=False, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
               name=name+'_bn1', lr_mult=lr_mult, reuse=reuse)

    if projection:
        shortcut = Conv(x, num_filter=num_filter, kernel=(1, 1), stride=(stride,)*2,
                        pad=(0, 0), no_bias=True, name=name+'_sc', lr_mult=lr_mult, reuse=reuse)
    else:
        shortcut = data

    if bottle_neck:
        x = Conv(x, num_filter=int(num_filter/4.), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                 no_bias=True, name=name+'_conv1', lr_mult=lr_mult, reuse=reuse)
        x = BNRelu(x, fix_gamma=False, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
                   name=name+'_bn2', lr_mult=lr_mult, reuse=reuse)
        x = Conv(x, num_filter=int(num_filter/4.), kernel=(3, 3), stride=(stride,)*2, pad=(dilate,)*2,
                 dilate=(dilate,)*2, no_bias=True, name=name+'_conv2', lr_mult=lr_mult, reuse=reuse)
        x = BNRelu(x, fix_gamma=False, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
                   name=name+'_bn3', lr_mult=lr_mult, reuse=reuse)
        x = Conv(x, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                 no_bias=True, name=name+'_conv3', lr_mult=lr_mult, reuse=reuse)
    else:
        x = Conv(x, num_filter=num_filter, kernel=(3, 3), stride=(stride,)*2, pad=(dilate,)*2,
                 dilate=(dilate,)*2, no_bias=True, name=name+'_conv1', lr_mult=lr_mult, reuse=reuse)
        x = BNRelu(x, fix_gamma=False, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
                   name=name+'_bn2', lr_mult=lr_mult, reuse=reuse)
        x = Conv(x, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                 no_bias=True, name=name+'_conv2', lr_mult=lr_mult, reuse=reuse)

    x = x + shortcut
    return x

def ResBlock(data, num_unit, num_filter, stride, dilate, bottle_neck, momentum=0.9, eps=1e-5,
             use_global_stats=False, name=None, lr_mult=1, reuse=None):
    assert name is not None
    x = ResUnit(data, num_filter, stride, dilate, True, bottle_neck, momentum, eps,
                use_global_stats, name+'_unit1', lr_mult, reuse)
    for i in range(1, num_unit):
        x = ResUnit(x, num_filter, 1, dilate, False, bottle_neck, momentum, eps,
                    use_global_stats, name+'_unit%d'%(i+1), lr_mult, reuse)
    return x

def _Resnet(x, num_units, num_filters, bottle_neck, momentum=0.9, eps=1e-5, use_global_stats=False, bn_data=True,
            strides=(1, 2, 2, 2), dilates=(1, 1, 1, 1), name=None, lr_mult=1, reuse=None):
    name = '' if name is None else name

    x = ResStem(x, num_filters[0], momentum, eps, use_global_stats, bn_data, name, lr_mult, reuse)
    for i in range(4):
        x = ResBlock(x, num_units[i], num_filters[i+1], strides[i], dilates[i], bottle_neck, 
                     momentum, eps, use_global_stats, name+'stage%d'%(i+1), lr_mult, reuse)
    x = BNRelu(x, fix_gamma=False, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
               name=name+'bn1', lr_mult=lr_mult, reuse=reuse)
    return x

def resnet18(x, momentum=0.9, eps=1e-5, use_global_stats=False, name=None, lr_mult=1, reuse=None):
    name = '' if name is None else name
    x = _Resnet(x, (2, 2, 2, 2), (64, 64, 128, 256, 512), False, momentum, eps, use_global_stats,
                name=name, lr_mult=lr_mult, reuse=reuse)
    x = Pool(x, (1, 1), pool_type='avg', global_pool=True)
    x = Flatten(x)
    return x

def resnet34(x, momentum=0.9, eps=1e-5, use_global_stats=False, name=None, lr_mult=1, reuse=None):
    name = '' if name is None else name
    x = _Resnet(x, (3, 4, 6, 3), (64, 64, 128, 256, 512), False, momentum, eps, use_global_stats,
                name=name, lr_mult=lr_mult, reuse=reuse)
    x = Pool(x, (1, 1), pool_type='avg', global_pool=True)
    x = Flatten(x)
    return x

def resnet50(x, momentum=0.9, eps=1e-5, use_global_stats=False, name=None, lr_mult=1, reuse=None):
    name = '' if name is None else name
    x = _Resnet(x, (3, 4, 6, 3), (64, 256, 512, 1024, 2048), True, momentum, eps, use_global_stats,
                name=name, lr_mult=lr_mult, reuse=reuse)
    x = Pool(x, (1, 1), pool_type='avg', global_pool=True)
    x = Flatten(x)
    return x

def resnet101(x, momentum=0.9, eps=1e-5, use_global_stats=False, name=None, lr_mult=1, reuse=None):
    name = '' if name is None else name
    x = _Resnet(x, (3, 4, 23, 3), (64, 256, 512, 1024, 2048), True, momentum, eps, use_global_stats,
                name=name, lr_mult=lr_mult, reuse=reuse)
    x = Pool(x, (1, 1), pool_type='avg', global_pool=True)
    x = Flatten(x)
    return x

def resnet50_largefov(x, num_cls, momentum=0.9, eps=1e-5, use_global_stats=False, name=None, lr_mult=10, reuse=None):
    name = '' if name is None else name
    x = _Resnet(x, (3, 4, 6, 3), (64, 256, 512, 1024, 2048), True, momentum, eps, use_global_stats,
                strides=(1, 2, 1, 1), dilates=(1, 1, 2, 4), name=name, lr_mult=1, reuse=reuse)
    x = Conv(x, num_cls, kernel=(3, 3), dilate=(12, 12), pad=(12, 12), name=name+'fc1', lr_mult=lr_mult, reuse=reuse)
    return x

def resnet101_largefov(x, num_cls, momentum=0.9, eps=1e-5, use_global_stats=False, name=None, lr_mult=10, reuse=None):
    name = '' if name is None else name
    x = _Resnet(x, (3, 4, 23, 3), (64, 256, 512, 1024, 2048), True, momentum, eps, use_global_stats,
                strides=(1, 2, 1, 1), dilates=(1, 1, 2, 4), name=name, lr_mult=1, reuse=reuse)
    x = Conv(x, num_cls, kernel=(3, 3), dilate=(12, 12), pad=(12, 12), name=name+'fc1', lr_mult=lr_mult, reuse=reuse)
    return x

'''
def resnet101_aspp(x, num_cls, momentum=0.9, eps=1e-5, use_global_stats=False, name=None, lr_mult=10, reuse=None):
    name = '' if name is None else name
    x = _Resnet(x, (3, 4, 23, 3), (64, 256, 512, 1024, 2048), True, momentum, eps, use_global_stats,
                strides=(1, 2, 1, 1), dilates=(1, 1, 2, 4), name=name, lr_mult=1, reuse=reuse)
    x_aspp = []
    for d in (6, 12, 18, 24):
        x_aspp.append(Conv(x, num_cls, kernel=(3, 3), dilate=(d, d), pad=(d, d),
            name=name+'fc1_aspp%d' % d, lr_mult=lr_mult, reuse=reuse))
    x = sum(x_aspp)
    return x
'''
