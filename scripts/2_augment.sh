python augment.py --name cifar10 --gpu 0 \
    --dataset cifar10 \
    --batch_size 64 \
    --genotype "Genotype(normal=[[('sep_conv_3x3', 1), ('skip_connect', 0)], [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_3x3', 0), ('sep_conv_3x3', 4)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('skip_connect', 2), ('max_pool_3x3', 0)], [('max_pool_3x3', 1), ('max_pool_3x3', 0)], [('skip_connect', 3), ('skip_connect', 2)]], reduce_concat=range(2, 6))"