from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'dil_sep_conv_5x5',
    'sep_conv_5x5',
]

REDUCE_PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'dil_sep_conv_5x5',
    'sep_conv_5x5',
]

op_level = {'skip_connect': 0,
            'max_pool_3x3': 0,
            'avg_pool_3x3': 0,
            'dep_dil_conv_3x3': 0,
            'conv_1x1': 0,
            'dep_conv_3x3': 0,
            'dil_conv_3x3': 1,
            'sep_conv1_3x3': 1,
            'dil_conv_5x5': 2,
            'dil_sep_conv_3x3': 2,
            'sep_conv_3x3': 2,
            'sep_conv1_5x5': 2,
            'dil_sep_conv_5x5': 3,
            'sep_conv_5x5': 3}

# op_graph = {'sep_conv_5x5': ['sep_conv_5x5', 'dil_conv_5x5', 'sep_conv_3x3', 'sep_conv1_5x5'],
#             'conv_5x5': ['conv_5x5', 'sep_conv_3x3', 'sep_conv1_5x5', 'conv_3x3'],
#             'dil_conv_5x5': ['dil_conv_5x5', 'dil_conv_3x3'],
#             'sep_conv_3x3': ['sep_conv_3x3', 'dil_conv_3x3', 'sep_conv1_3x3'],
#             'sep_conv1_5x5': ['sep_conv1_5x5', 'dil_conv_3x3', 'sep_conv1_3x3'],
#             'conv_3x3': ['conv_3x3', 'sep_conv1_3x3'],
#             'dil_conv_3x3': ['dil_conv_3x3'],
#             'sep_conv1_3x3': ['sep_conv1_3x3'],
#             'skip_connect': ['skip_connect'],
#             'max_pool_3x3': ['max_pool_3x3'],
#             'avg_pool_3x3': ['avg_pool_3x3'],
#             }

# op_graph = {'sep_conv_5x5': ['sep_conv_5x5', 'sep_conv_3x3', 'dil_conv_5x5'],
#             'dil_conv_3x3': ['sep_conv_3x3', 'dil_conv_5x5', 'dil_conv_3x3'],
#             'skip_connect': ['skip_connect'],
#             'max_pool_3x3': ['max_pool_3x3'],
#             'avg_pool_3x3': ['avg_pool_3x3'],
#             }

op_graph = {'dil_sep_conv_5x5': ['dil_sep_conv_5x5', 'dil_conv_5x5', 'dil_sep_conv_3x3', 'sep_conv1_5x5'],
            'sep_conv_5x5': ['sep_conv_5x5', 'dil_sep_conv_3x3', 'sep_conv1_5x5', 'sep_conv_3x3'],
            'dil_conv_5x5': ['dil_conv_5x5', 'dil_conv_3x3'],
            'dil_sep_conv_3x3': ['dil_sep_conv_3x3', 'dil_conv_3x3', 'sep_conv1_3x3'],
            'sep_conv1_5x5': ['sep_conv1_5x5', 'dil_conv_3x3', 'sep_conv1_3x3'],
            'sep_conv_3x3': ['sep_conv_3x3', 'sep_conv1_3x3'],
            'dil_conv_3x3': ['dil_conv_3x3', 'skip_connect', 'max_pool_3x3', 'avg_pool_3x3'],
            'sep_conv1_3x3': ['sep_conv1_3x3', 'skip_connect', 'max_pool_3x3', 'avg_pool_3x3'],
            'skip_connect': ['skip_connect'],
            'max_pool_3x3': ['max_pool_3x3'],
            'avg_pool_3x3': ['avg_pool_3x3'],
            }

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

PDARTS = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

geno01 = Genotype(
    normal=[('conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('conv_5x5', 1), ('skip_connect', 0),
            ('sep_conv_5x5', 3), ('conv_5x5', 0), ('conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

geno01_com = Genotype(
    normal=[('conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv1_5x5', 3), ('conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv1_5x5', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

geno01_server = Genotype(
    normal=[('conv_5x5', 1), ('conv_5x5', 0), ('skip_connect', 0), ('conv_5x5', 1), ('skip_connect', 0),
            ('conv_5x5', 1), ('conv_5x5', 1), ('conv_5x5', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

geno01_server_com = Genotype(
    normal=[('sep_conv1_5x5', 1), ('sep_conv1_5x5', 0), ('skip_connect', 0), ('sep_conv1_5x5', 1), ('skip_connect', 0),
            ('sep_conv1_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv1_5x5', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

geno02_server = Genotype(
    normal=[('sep_conv_5x5', 0), ('conv_5x5', 1), ('sep_conv_5x5', 0), ('conv_5x5', 2), ('conv_5x5', 3),
            ('sep_conv_5x5', 0), ('conv_5x5', 4), ('conv_5x5', 3)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2),
            ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

geno02_server_com = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv1_5x5', 3),
            ('sep_conv_3x3', 0), ('sep_conv1_5x5', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2),
            ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

geno03_server = Genotype(
    normal=[('sep_conv_5x5', 0), ('conv_5x5', 1), ('conv_5x5', 0), ('conv_5x5', 2), ('skip_connect', 1),
            ('conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

geno03_server_com = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 1),
            ('conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))


