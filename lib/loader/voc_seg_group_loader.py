from ..utils import *
from .voc_seg_loader import load_batch_semantic


class VOCSegGroupLoader(mx.io.DataIter):
    def __init__(self, image_root, label_root, annotation_root, data_list,
            batch_size, group_size, num_block, target_size,
            pad=False, shuffle=False, rand_scale=False, rand_mirror=False, rand_crop=False):

        assert group_size >= 2, "'group_size': # common-class images, typical value is 2 for pairs"
        assert num_block >= 1,  "'num_block':  should equal # GPU"
        assert batch_size % (group_size * num_block) == 0

        with open(data_list, 'r') as f:
            data_names = [x.strip() for x in f.readlines()]

        if pad and (len(data_names) % batch_size > 0):
            pad_num = batch_size - (len(data_names) % batch_size)
            data_names = data_names + data_names[:pad_num]

        self.image_src_list = [os.path.join(image_root, x+'.jpg') for x in data_names]
        self.label_src_list = [os.path.join(label_root, x+'.png') for x in data_names] \
                if label_root is not None else [None] * len(data_names)
        self.ann_list = [VOC.get_annotation(os.path.join(annotation_root, x+'.xml')) for x in data_names]

        self.batch_size = batch_size
        self.group_size = group_size
        self.num_block = num_block
        self.meta_length = self.batch_size // (self.num_block * self.group_size)

        self.target_size = target_size
        self.shuffle = shuffle
        self.rand_scale = rand_scale
        self.rand_mirror = rand_mirror
        self.rand_crop = rand_crop

        scale_pool = [0.5, 0.75, 1, 1.25, 1.5]
        self.scale_sampler = lambda : np.random.choice(scale_pool)

        self.index = list(range(len(data_names)))
        self.num_batch = len(data_names) // self.batch_size
        self.reset()

    def reset(self):
        self.index_pointer = 0
        self.cache = []
        if self.shuffle:
            np.random.shuffle(self.index)
    
    def pop(self):
        if len(self.cache) > 0:
            index = self.cache.pop()
        elif self.index_pointer < len(self.index):
            index = self.index[self.index_pointer]
            self.index_pointer += 1
        else:
            raise StopIteration
        return index

    def is_ok(self, a, b):
        lbl_a = self.ann_list[a]
        lbl_b = self.ann_list[b]
        return len(set(lbl_a) - set(lbl_b)) < len(lbl_a)

    def next(self):
        indices = []
        while len(indices) < self.batch_size // self.group_size:
            cache = []
            partners = [self.pop()]
            while len(partners) < self.group_size:
                this = self.pop()
                while not all([self.is_ok(prev, this) for prev in partners]):
                    cache.append(this)
                    this = self.pop()
                partners.append(this)
            indices.append(partners)
            self.cache = cache[::-1] + self.cache

        indices = sum( [sum(zip(*indices[i : i+self.meta_length]), tuple()) \
                for i in range(0, len(indices), self.meta_length)], tuple() )

        image_src_list = [self.image_src_list[i] for i in indices]
        label_src_list = [self.label_src_list[i] for i in indices]
        self.cache_image_src_list = image_src_list

        batch = load_batch_semantic(image_src_list, label_src_list, self.target_size, self.scale_sampler,
                                    self.rand_scale, self.rand_mirror, self.rand_crop)
        return batch

