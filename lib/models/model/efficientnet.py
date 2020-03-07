from .layers import *
from .layers_custom import *
import re
import numpy as np
from collections import namedtuple

DEFAULT_EFFICIENT_PARAMS = {
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5)
}

DEFAULT_EFFICIENT_BLOCK_ARGS = [
    'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
]

def config_efficientnet(model_name):
    assert re.match(r'^efficientnet-b[0-7]$', model_name), model_name
    
    #efficientnet_params = {
    #    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    #    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    #    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    #    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    #    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    #    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    #    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    #    'efficientnet-b7': (2.0, 3.1, 600, 0.5)
    #}[model_name]
    efficientnet_params = DEFAULT_EFFICIENT_PARAMS[model_name]

    #block_args = [
    #    'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    #    'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    #    'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
    #    'r1_k3_s11_e6_i192_o320_se0.25',
    #]
    block_args = DEFAULT_EFFICIENT_BLOCK_ARGS

    #block_args = [
    #    'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    #    'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    #    'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s11_e6_i112_o192_se0.25_d2',
    #    'r1_k3_s11_e6_i192_o320_se0.25_d2',
    #]

    width_coefficient, depth_coefficient, resolution, dropout_rate = efficientnet_params
    global_params = {
        'block_args': block_args,
        'batch_norm_momentum': 0.99,
        'batch_norm_epsilon': 1e-3,
        'dropout_rate': dropout_rate,
        'survival_prob': 0.8,
        'num_classes': 1000,
        'width_coefficient': width_coefficient,
        'depth_coefficient': depth_coefficient,
        'depth_divisor': 8,
        'min_depth': None,
        'use_se': True,
        'clip_projection_output': False
        # data_format, relu_fn, batch_norm
    }
    global_params = namedtuple('global_parmas', sorted(global_params))(**global_params)

    kv_list = [dict([re.split(r'([\d\.]+)', op)[:2] for op in _block_args.split('_')]) for _block_args in block_args]
    block_args_list = [{
        'kernel_size': int(kv['k']),
        'num_repeat': int(kv['r']),
        'input_filters': int(kv['i']),
        'output_filters': int(kv['o']),
        'expand_ratio': int(kv['e']),
        'id_skip': 'noskip' not in block_string,
        'se_ratio': float(kv['se']) if 'se' in kv else None,
        'strides': (int(kv['s'][0]), int(kv['s'][1])),
        'conv_type': int(kv.get('c', '0')),
        'fused_conv': int(kv.get('f', '0')),
        'super_pixel': int(kv.get('p', '0')),
        'dilate': int(kv.get('d', '1')),
        'condconv': 'cc' in block_string,
        'survival_prob': 1.0
    } for kv, block_string in zip(kv_list, block_args)]
    block_args_list = [namedtuple('block_args', sorted(x))(**x) for x in block_args_list]

    return block_args_list, global_params

def MBConvBlock(data, block_args, global_params, use_global_stats, block_id, name, lr_mult, reuse, input_size=None):
    if block_args.super_pixel:
        raise NotImplementedError
    if block_args.condconv:
        raise NotImplementedError

    kernel = (block_args.kernel_size,)*2
    dilate = (1 if kernel[0] == 1 else block_args.dilate,)*2
    pad = ( ((kernel[0]-1)*dilate[0]+1)//2, )*2
    #pad = (kernel[0]//2,)*2
    #dilate = (1, 1)
    momentum = global_params.batch_norm_momentum
    eps = global_params.batch_norm_epsilon

    num_filters = block_args.input_filters * block_args.expand_ratio

    conv_id, bn_id = 0, 0

    if block_args.fused_conv:
        x = Conv(data, num_filters, kernel, block_args.strides, pad=pad, dilate=dilate,
                no_bias=True, name=name+'block%d_conv'%block_id, lr_mult=lr_mult, reuse=reuse)
        #x = Conv(data, num_filters, kernel, block_args.strides, pad='same',
        #        no_bias=True, name=name+'block%d_conv'%block_id, lr_mult=lr_mult, reuse=reuse, input_size=input_size)
    else:
        if block_args.expand_ratio != 1:
            x = Conv(data, num_filters, (1,1), no_bias=True, name=name+'block%d_conv%d'%(block_id, conv_id), lr_mult=lr_mult, reuse=reuse)
            x = BN(x, momentum=momentum, eps=eps, use_global_stats=use_global_stats, name=name+'block%d_bn%d'%(block_id, bn_id), lr_mult=lr_mult, reuse=reuse)
            x = Swish(x)
            conv_id, bn_id = conv_id + 1, bn_id + 1
        else:
            x = data

        x = Conv(x, num_filters, kernel, block_args.strides, pad=pad, dilate=dilate,
                num_group=num_filters, no_bias=True, name=name+'block%d_depthwise_conv0'%block_id, lr_mult=lr_mult, reuse=reuse)
        #x = Conv(x, num_filters, kernel, block_args.strides, pad='same', 
        #        num_group=num_filters, no_bias=True, name=name+'block%d_depthwise_conv0'%block_id, lr_mult=lr_mult, reuse=reuse, input_size=input_size)

    x = BN(x, momentum=momentum, eps=eps, use_global_stats=use_global_stats, name=name+'block%d_bn%d'%(block_id, bn_id), lr_mult=lr_mult, reuse=reuse)
    x = Swish(x)
    bn_id += 1

    has_se = global_params.use_se and block_args.se_ratio is not None and 0 < block_args.se_ratio <= 1
    if has_se:
        num_filters_rd = max(1, int(block_args.input_filters * block_args.se_ratio))
        x_se = mx.sym.mean(x, axis=(2, 3), keepdims=True)
        x_se = Conv(x_se, num_filters_rd, (1,1), name=name+'block%d_se_conv0'%block_id, lr_mult=lr_mult, reuse=reuse)
        x_se = Swish(x_se)
        x_se = Conv(x_se, num_filters, (1,1), name=name+'block%d_se_conv1'%block_id, lr_mult=lr_mult, reuse=reuse)
        x = mx.sym.broadcast_mul(mx.sym.sigmoid(x_se), x)

    x = Conv(x, block_args.output_filters, (1,1), no_bias=True, name=name+'block%d_conv%d'%(block_id, conv_id), lr_mult=lr_mult, reuse=reuse)
    x = BN(x, momentum=momentum, eps=eps, use_global_stats=use_global_stats, name=name+'block%d_bn%d'%(block_id, bn_id), lr_mult=lr_mult, reuse=reuse)
    conv_id, bn_id = conv_id + 1, bn_id + 1

    if global_params.clip_projection_output:
        x = mx.sym.clip(x, a_min=-6, a_max=6)

    if block_args.id_skip and all([s == 1 for s in block_args.strides]) and block_args.input_filters == block_args.output_filters:
        if block_args.survival_prob > 0:
            x = mx.sym.Custom(x, p=1-block_args.survival_prob, op_type='DropConnect')
        x = x + data
    
    return x

def MBConvBlockWithoutDepthwise(data, block_args, global_params, use_global_stats, begin_id, name, lr_mult, reuse):
    raise NotImplementedError

def meta_efficientnet(model_name, get_internals=False, input_size=None):
    block_args_list, global_params = config_efficientnet(model_name)

    def round_filters(num_filters):
        multiplier = global_params.width_coefficient
        if not multiplier:
            return num_filters
        divisor = global_params.depth_divisor
        min_depth = global_params.min_depth

        num_filters = num_filters * multiplier
        new_num = max(min_depth or divisor, int(num_filters+divisor/2)//divisor*divisor)
        if new_num < 0.9 * num_filters:
            new_num += divisor
        return int(new_num)

    def round_repeats(num_repeat):
        multiplier = global_params.depth_coefficient
        if not multiplier:
            return num_repeat
        return int(np.ceil(multiplier * num_repeat))

    def efficient_model(data, use_global_stats=False, bn_data=False, name=None, lr_mult=1, reuse=None):
        name = '' if name is None else name
        endpoints = {}

        momentum = global_params.batch_norm_momentum
        eps = global_params.batch_norm_epsilon
        endpoints['input'] = data

        # data
        if bn_data:
            data = BN(data, fix_gamma=True, momentum=momentum, eps=eps, name='bn_data', lr_mult=lr_mult, reuse=reuse)

        # Stem
        x = Conv(data, round_filters(32), (3,3), (2,2), pad=(1,1), no_bias=True, name=name+'stem_conv0', lr_mult=lr_mult, reuse=reuse)
        #x = Conv(data, round_filters(32), (3,3), (2,2), pad='same', no_bias=True, name=name+'stem_conv0', lr_mult=lr_mult, reuse=reuse, input_size=input_size)
        x = BN(x, momentum=momentum, eps=eps, use_global_stats=use_global_stats, name='stem_bn0', lr_mult=lr_mult, reuse=reuse)
        x = Swish(x)
        endpoints['stem'] = x

        # Blocks
        block_id = 0
        total_blocks = sum([block_args.num_repeat for block_args in block_args_list])
        survival_prob = global_params.survival_prob

        for i, block_args in enumerate(block_args_list):
            assert block_args.num_repeat > 0
            assert block_args.super_pixel in [0, 1, 2]
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters),
                output_filters=round_filters(block_args.output_filters),
                num_repeat=round_repeats(block_args.num_repeat),
                survival_prob=1.0 - (1.0-global_params.survival_prob)*float(block_id)/total_blocks
            )

            ConvBlock = {0: MBConvBlock, 1: MBConvBlockWithoutDepthwise}[block_args.conv_type]

            # the first block
            x = ConvBlock(x, block_args, global_params, use_global_stats=use_global_stats, block_id=block_id, name=name, lr_mult=lr_mult, reuse=reuse)
            endpoints['block%d'%block_id] = x
            block_id += 1

            # the following blocks
            for j in range(block_args.num_repeat - 1):
                block_args = block_args._replace(
                    input_filters=block_args.output_filters,
                    strides=(1, 1),
                    survival_prob=1.0 - (1.0-global_params.survival_prob)*float(block_id)/total_blocks
                )
                x = ConvBlock(x, block_args, global_params, use_global_stats=use_global_stats, block_id=block_id, name=name, lr_mult=lr_mult, reuse=reuse, input_size=input_size)
                endpoints['block%d'%block_id] = x
                block_id += 1

        # Head
        x = Conv(x, round_filters(1280), (1,1), no_bias=True, name=name+'head_conv0', lr_mult=lr_mult, reuse=reuse)
        x = BN(x, momentum=momentum, eps=eps, use_global_stats=use_global_stats, name='head_bn0', lr_mult=lr_mult, reuse=reuse)
        x = Swish(x)
        if global_params.dropout_rate > 0:
            x = Drop(x, p=global_params.dropout_rate)
        endpoints['head'] = x

        x = Pool(x, kernel=(1, 1), pool_type='avg', global_pool=True)
        x = mx.sym.flatten(x)
        x = FC(x, global_params.num_classes, name=name+'head_fc0', lr_mult=lr_mult, reuse=reuse)
        endpoints['logit'] = x
        x = mx.sym.softmax(x, axis=1)
        endpoints['prob'] = x

        if get_internals:
            return x, endpoints
        return x

    return efficient_model

efficientnet_b0 = meta_efficientnet('efficientnet-b0')


def tf2mx_params(ckpt_file, dst_file=None, name='', use_ema=True):
    convert_w = lambda x: mx.nd.array(x.transpose(3, 2, 0, 1) if x.ndim == 4 else x.T)
    convert_b = lambda x: mx.nd.array(x)
    convert_dp_w = lambda x: mx.nd.array(x.transpose(2, 3, 0, 1))
    lookup_ptype = {'kernel': ('arg', 'weight', convert_w), 'bias': ('arg', 'bias', convert_b),
    'depthwise_kernel': ('arg', 'weight', convert_dp_w),
    'gamma': ('arg', 'gamma', convert_b), 'beta': ('arg', 'beta', convert_b),
    'moving_mean': ('aux', 'moving_mean', convert_b), 'moving_variance': ('aux', 'moving_var', convert_b)}
    lookup_op = {'conv2d': 'conv', 'depthwise_conv2d': 'depthwise_conv',
    'tpu_batch_normalization': 'bn', 'dense': 'fc'}

    def mapKey(tf_key):
        names = tf_key.split('/')
        if not re.match(r'^efficientnet-b[0-7]$', names[0]):
            return None, None
        block, op, ptype = names[1:4]

        if block.startswith('blocks'):
            block = 'block' + block.split('_')[-1]
        block_name = name + block + '_'

        if op == 'se':
            op, ptype = names[3:5]
            block_name = block_name + 'se_'

        r = re.match(r'^\w*_(\d+)$', op)
        op_id = (r.group(1) if r else '0')
        _op = re.match(r'^(\w+)_\d+$', op).group(1) if r else op
        try:
            prefix, suffix, converter = lookup_ptype[ptype]
        except:
            raise KeyError("[{}], ({}, {}, {}), {}".format(ptype, block, op, ptype, tf_key))

        op_name = lookup_op[_op]
        return prefix + ':' + block_name+op_name+op_id + '_' + suffix, converter

    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        import tensorflow  as tf
    reader = tf.train.load_checkpoint(ckpt_file)
    shape_map = reader.get_variable_to_shape_map()
    keys = sorted(shape_map.keys())

    ema_keys = [k for k in keys if k.endswith('ExponentialMovingAverage')]
    keys = list(set(list(set(keys) - set(ema_keys)) + [k.rsplit('/', 1)[0] for k in ema_keys]))
    keys_ = [k + '/ExponentialMovingAverage' for k in keys]
    kk = {k: k_ if (use_ema and (k_ in ema_keys)) else k for k, k_ in zip(keys, keys_)}

    mx_params = {}
    for k in kk.keys():
        tf_key = kk[k]
        mx_key, converter = mapKey(k)
        if mx_key is None:
            if tf_key != 'global_step':
                print("Cannot parse tf_key: %s" % tf_key)
            continue
        if mx_key in mx_params:
            raise KeyError("Duplicate key: %s, %s, %s" % (k, tf_key, mx_key))
        mx_params[mx_key] = converter(reader.get_tensor(tf_key))

    if dst_file is not None:
        mx.nd.save(dst_file, mx_params)

    arg_params = {k[4:]: v for k, v in mx_params.items() if k.startswith('arg:')}
    aux_params = {k[4:]: v for k, v in mx_params.items() if k.startswith('aux:')}
    return arg_params, aux_params
