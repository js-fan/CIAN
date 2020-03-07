from .resnet import _Resnet
from .cian_modules import build_self_affinity, build_cross_affinity
from .layers import *
from .multi_scale import *


def resnet101_largefov_SA(x, num_cls, is_downsample=True,
        in_embed_type='conv', out_embed_type='convbn', sim_type='dot',
        use_global_stats_backbone=False, use_global_stats_affinity=False,
        lr_mult=10, reuse=None, **kwargs):

    x_raw = _Resnet(x, (3, 4, 23, 3), (64, 256, 512, 1024, 2048), True,
            use_global_stats=use_global_stats_backbone,
            strides=(1, 2, 1, 1), dilates=(1, 1, 2, 4), lr_mult=1, reuse=reuse)

    x_res = build_self_affinity(x_raw, 1024, 2048, is_downsample,
            in_embed_type, out_embed_type, sim_type,
            use_global_stats_affinity, lr_mult, reuse)

    x = x_raw + x_res
    x = Conv(x, num_cls, (3, 3), (1, 1), dilate=(12, 12), pad=(12, 12),
            no_bias=True, name='fc1', lr_mult=lr_mult, reuse=reuse)
    return x

def resnet101_largefov_CA(x, num_cls, is_downsample=True,
        in_embed_type='conv', out_embed_type='convbn', sim_type='dot',
        group_size=2, merge_type='max', merge_self=True,
        use_global_stats_backbone=False, use_global_stats_affinity=False,
        lr_mult=10, reuse=None):

    x_raw = _Resnet(x, (3, 4, 23, 3), (64, 256, 512, 1024, 2048), True,
            use_global_stats=use_global_stats_backbone,
            strides=(1, 2, 1, 1), dilates=(1, 1, 2, 4), lr_mult=1, reuse=reuse)

    x_res_self, x_res_cross = build_cross_affinity(x_raw, 1024, 2048, is_downsample,
            in_embed_type, out_embed_type, sim_type,
            group_size, merge_type, merge_self,
            use_global_stats_affinity, lr_mult, reuse)

    x_self = x_raw + x_res_self
    x_self = Conv(x_self, num_cls, (3, 3), (1, 1), dilate=(12, 12), pad=(12, 12),
            no_bias=True, name='fc1', lr_mult=lr_mult, reuse=reuse)
    x_cross = x_raw + x_res_cross
    x_cross = Conv(x_cross, num_cls, (3, 3), (1, 1), dilate=(12, 12), pad=(12, 12),
            no_bias=True, name='fc1', lr_mult=lr_mult, reuse=x_self)
    return x_self, x_cross

