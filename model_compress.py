import numpy as np
import torch.nn.functional as F
from operations import *
from genotypes import op_graph

class CompressedMixedOp(nn.Module):

    def __init__(self, C, stride, compressing_op):
        super(CompressedMixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        # self.operations = PRIMITIVES
        self.operations = op_graph[compressing_op]
        for i in range(len(self.operations)):
            op_name = self.operations[i]
            op = OPS[op_name](C, stride, False)
            self.m_ops.append(op)

        self.op_count = len(self.m_ops)

    def forward(self, x, weights):
        # keep previous operation: weights = [1.0]

        return sum(w * op(x) for w, op in zip(weights, self.m_ops))

class CompressedCell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, cell_name, reduction_prev, layer_num=None,
                 residual_connection=False):
        super(CompressedCell, self).__init__()
        self.cell_name = cell_name
        self.residual_connection = residual_connection
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        op_names, indices = zip(*getattr(genotype, cell_name))
        concat = getattr(genotype, cell_name + '_concat')

        multiplier = len(genotype.reduce_concat)

        ##########################################
        if self.residual_connection:
            if layer_num == 0:
                self.skipconn = nn.Sequential(Conv1x1(C_prev, multiplier * C, 1), nn.BatchNorm2d(multiplier * C))
            elif cell_name == 'reduce':
                self.skipconn = nn.Sequential(Conv1x1(C_prev, multiplier * C, 2), nn.BatchNorm2d(multiplier * C))
            else:
                self.skipconn = Identity()
        ##########################################

        self._compile(C, op_names, indices, concat, cell_name)

    def _compile(self, C, op_names, indices, concat, cell_name):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if cell_name == 'reduce' and index < 2 else 1
            # op = OPS[name](C, stride, True)
            op = CompressedMixedOp(C, stride, name)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, weights):
        original_input = s1

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1, weights[2 * i])
            h2 = op2(h2, weights[2 * i + 1])

            s = h1 + h2
            states += [s]

        if self.residual_connection:
            return torch.cat([states[i] for i in self._concat], dim=1) + self.skipconn(original_input)
        else:
            return torch.cat([states[i] for i in self._concat], dim=1)


class CompressedNetwork(nn.Module):

    def __init__(self, genotype, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3,
                 residual_connection=False):
        super(CompressedNetwork, self).__init__()
        self.genotype = genotype
        self.cell_names = ['normal', 'reduce']
        self.param_names = ['alphas_normal', 'alphas_reduce']
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        # self.cells = nn.ModuleList()
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                cell_name = 'reduce'
                reduction = True
            else:
                reduction = False
                cell_name = 'normal'

            cell = CompressedCell(genotype, C_prev_prev, C_prev, C_curr, cell_name, reduction_prev, layer_num=i, residual_connection=residual_connection)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.num_paths = len(genotype.normal)
        self._initialize_alphas()

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            weights = []

            param_name = self.param_names[self.cell_names.index(cell.cell_name)]
            tmp_param = getattr(self, param_name)
            for j in range(self.num_paths):
                if len(tmp_param[j]) == 1:
                    weights.append(tmp_param[j])
                else:
                    weights.append(F.softmax(tmp_param[j], dim=0))

            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):

        self.alphas_normal = []
        op_names, indices = zip(*getattr(self.genotype, 'normal'))
        for name in op_names:
            num_ops = len(op_graph[name])
            self.alphas_normal.append(nn.Parameter(torch.cuda.FloatTensor(1e-3 * np.random.randn(num_ops))))

        self.alphas_reduce = []
        op_names, indices = zip(*getattr(self.genotype, 'reduce'))
        for name in op_names:
            num_ops = len(op_graph[name])
            self.alphas_reduce.append(nn.Parameter(torch.cuda.FloatTensor(1e-3 * np.random.randn(num_ops))))

    def arch_parameters(self):
        self._arch_parameters = []
        for params in [self.alphas_normal, self.alphas_reduce]:
            for i in range(len(params)):
                self._arch_parameters.append(params[i])
        # print(self._arch_parameters)
        return self._arch_parameters
