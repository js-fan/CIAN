from .layers import *

def in_embedding_conv(x_feat, num_filter_hidden, is_downsample=True, lr_mult=1, reuse=None):
    x_query = Conv(x_feat, num_filter_hidden, (1, 1), no_bias=True,
            name='conv_embed_q', lr_mult=lr_mult, reuse=reuse)
    x_key = Conv(x_feat, num_filter_hidden, (1, 1), no_bias=True,
            name='conv_embed_k', lr_mult=lr_mult, reuse=reuse)
    x_value = Conv(x_feat, num_filter_hidden, (1, 1), no_bias=True,
            name='conv_embed_v', lr_mult=lr_mult, reuse=reuse)
    
    if is_downsample:
        x_key =   Pool(x_key, (3, 3), (2, 2), (1, 1))
        x_value = Pool(x_value, (3, 3), (2, 2), (1, 1))
    return x_query, x_key, x_value

def out_embedding_convbn(x_res, num_filter_out, use_global_stats=False, lr_mult=1, reuse=None):
    x_res = Conv(x_res, num_filter_out, (1, 1), no_bias=True,
            name='conv_out', lr_mult=lr_mult, reuse=reuse)
    x_res = BN(x_res, fix_gamma=False, use_global_stats=use_global_stats,
            name='bn_out', lr_mult=lr_mult, reuse=reuse)
    return x_res

def compute_sim_mat(x_key, x_query, sim_type):
    if sim_type  == 'dot':
        sim_mat = mx.sym.batch_dot(x_key, x_query, transpose_a=True)
    elif sim_type == 'cosine':
        x_key_norm = mx.sym.L2Normalization(x_key, mode='channel')
        x_query_norm = mx.sym.L2Normalization(x_query, mode='channel')
        sim_mat = mx.sym.batch_dot(x_key_norm, x_query_norm, transpose_a=True)
    else:
        raise ValueError(sim_type)
    return sim_mat

def build_self_affinity(x_feat, num_filter_hidden, num_filter_out, is_downsample=True,
        in_embed_type='conv', out_embed_type='convbn', sim_type='dot',
        use_global_stats=False, lr_mult=1, reuse=None, return_internals=False):
    get_embedding_in =  eval('in_embedding_' + in_embed_type)
    get_embedding_out = eval('out_embedding_' + out_embed_type)

    x_query, x_key, x_value = get_embedding_in(x_feat,
            num_filter_hidden, is_downsample, lr_mult, reuse)

    x_query = mx.sym.reshape(x_query, (0, 0, -3))
    x_key   = mx.sym.reshape(x_key,   (0, 0, -3))
    x_value = mx.sym.reshape(x_value, (0, 0, -3))

    sim_mat = compute_sim_mat(x_key, x_query, sim_type)
    sim_mat = mx.sym.softmax(sim_mat, axis=1)

    x_res = mx.sym.batch_dot(x_value, sim_mat)
    x_res = mx.sym.reshape_like(x_res, x_feat, lhs_begin=2, lhs_end=3, rhs_begin=2, rhs_end=4)

    x_out = get_embedding_out(x_res, num_filter_out, use_global_stats, lr_mult, reuse)

    if return_internals:
        return x_out, x_query, x_key, x_value, sim_mat, x_res
    return x_out

def build_cross_affinity(x_feat, num_filter_hidden, num_filter_out, is_downsample=True,
        in_embed_type='conv', out_embed_type='convbn', sim_type='dot',
        group_size=2, merge_type='max', merge_self=True,
        use_global_stats=False, lr_mult=1, reuse=None):
    get_embedding_in =  eval('in_embedding_' + in_embed_type)
    get_embedding_out = eval('out_embedding_' + out_embed_type)

    x_out_self, x_query, x_key, x_value, sim_mat_self, x_res_self = build_self_affinity(
            x_feat, num_filter_hidden, num_filter_out, is_downsample,
            in_embed_type, out_embed_type, sim_type,
            use_global_stats, lr_mult, reuse, True)
    
    # split
    x_key_sp =   list(mx.sym.split(x_key, num_outputs=group_size, axis=0))
    x_value_sp = list(mx.sym.split(x_value, num_outputs=group_size, axis=0))

    # roll, res
    x_res_list = []
    for i in range(group_size - 1):
        x_key_sp = x_key_sp[1:] + x_key_sp[0:1]
        x_value_sp = x_value_sp[1:] + x_value_sp[0:1]

        x_key_roll = mx.sym.concat(*x_key_sp, dim=0)
        x_value_roll = mx.sym.concat(*x_value_sp, dim=0)

        sim_mat = compute_sim_mat(x_key_roll, x_query, sim_type)
        sim_mat = mx.sym.softmax(sim_mat, axis=1)

        x_res = mx.sym.batch_dot(x_value_roll, sim_mat)
        x_res = mx.sym.reshape_like(x_res, x_feat, lhs_begin=2, lhs_end=3, rhs_begin=2, rhs_end=4)
        x_res_list.append(x_res)

    # merge
    if merge_self:
        x_res_list.append(x_res_self)

    if merge_type == 'max':
        x_res_cross = x_res_list[0]
        for x_res in x_res_list[1:]:
            x_res_cross = mx.sym.maximum(x_res_cross, x_res)
    elif merge_type == 'avg':
        x_res_cross = sum(x_res_list) / len(x_res_list)

    # embed out
    x_out_cross = get_embedding_out(x_res_cross, num_filter_out, use_global_stats, lr_mult, x_out_self)

    return x_out_self, x_out_cross


