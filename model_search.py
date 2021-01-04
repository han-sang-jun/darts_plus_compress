import numpy as np
import torch.nn.functional as F
from operations import *
from genotypes import PRIMITIVES
from genotypes import REDUCE_PRIMITIVES


class MixedOp(nn.Module):

    def __init__(self, C, stride, reduction):
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        # self.operations = PRIMITIVES
        if reduction:
            self.operations = REDUCE_PRIMITIVES
        else:
            self.operations = PRIMITIVES
        for i in range(len(self.operations)):
            op_name = self.operations[i]
            op = OPS[op_name](C, stride, False)
            if 'pool' in op_name:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self.m_ops.append(op)

        self.op_count = len(self.m_ops)

    def forward(self, x, weights):
        # keep previous operation: weights = [1.0]

        return sum(w * op(x) for w, op in zip(weights, self.m_ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, layer_num=None,
                 residual_connection = False):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.residual_connection = residual_connection
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        ###########################
        if self.residual_connection:
            if layer_num == 0:
                self.skipconn = nn.Sequential(Conv1x1(C_prev, multiplier*C, 1), nn.BatchNorm2d(multiplier*C))
            elif reduction:
                self.skipconn = nn.Sequential(Conv1x1(C_prev, multiplier*C, 2), nn.BatchNorm2d(multiplier*C))
            else:
                self.skipconn = Identity()
        ###########################

        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                # j = starting node, i+2 = ending node
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, reduction)
                self.cell_ops.append(op)

        self.stride = 1

    def forward(self, s0, s1, weights):
        original_input = s1
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.cell_ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        if self.residual_connection:
            return torch.cat(states[-self._multiplier:], dim=1) + self.skipconn(original_input)
        else:
            return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3,
                 residual_connection=False):
        super(Network, self).__init__()
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
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                            residual_connection)
            else:
                reduction = False
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                            residual_connection)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.num_paths = sum(1 for i in range(self._steps) for n in range(2 + i))
        self._initialize_alphas()

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if self.alphas_reduce.size(1) == 1:
                    weights = F.softmax(self.alphas_reduce, dim=0)
                else:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                if self.alphas_normal.size(1) == 1:
                    weights = F.softmax(self.alphas_normal, dim=0)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        reduce_ops = len(REDUCE_PRIMITIVES)
        self.alphas_normal = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_ops)))
        self.alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, reduce_ops)))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

