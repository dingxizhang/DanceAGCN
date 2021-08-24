import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable



def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches, uniform=False):
    if uniform:
        nn.init.uniform_(conv.weight)
        nn.init.uniform_(conv.bias)
    else:
        weight = conv.weight
        n = weight.size(0)
        k1 = weight.size(1)
        k2 = weight.size(2)
        nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
        nn.init.constant_(conv.bias, 0)


def conv_init(conv, uniform=False):
    if uniform:
        nn.init.uniform_(conv.weight)
        nn.init.uniform_(conv.bias)
    else:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, bn=True, uniform_init=False):
        super(unit_tcn, self).__init__()
        # pad = int((kernel_size - 1) / 2)
        pad = round((kernel_size - 1) / 2)  # DM
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv, uniform=uniform_init)

        if bn:
            bn_init(self.bn, 1)  # DM

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, adaptive=True, attention=True,
                 bn=True, uniform_init=False):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        num_jpts = A.shape[-1]

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
            self.alpha = nn.Parameter(torch.zeros(1))
            # self.beta = nn.Parameter(torch.ones(1))
            # nn.init.constant_(self.PA, 1e-6)
            # self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
            # self.A = self.PA
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.adaptive = adaptive

        if attention:
            # self.beta = nn.Parameter(torch.zeros(1))
            # self.gamma = nn.Parameter(torch.zeros(1))
            # unified attention
            # self.Attention = nn.Parameter(torch.ones(num_jpts))

            # temporal attention
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)

            if uniform_init:
                nn.init.uniform_(self.conv_ta.weight)
                nn.init.uniform_(self.conv_ta.bias)
            else:
                nn.init.constant_(self.conv_ta.weight, 0)
                nn.init.constant_(self.conv_ta.bias, 0)

            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)

            if uniform_init:
                nn.init.uniform_(self.conv_sa.weight)
                nn.init.uniform_(self.conv_sa.bias)
            else:
                nn.init.xavier_normal_(self.conv_sa.weight)
                nn.init.constant_(self.conv_sa.bias, 0)

            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)

            if uniform_init:
                nn.init.uniform_(self.fc1c.weight)
                nn.init.uniform_(self.fc1c.bias)
                nn.init.uniform_(self.fc2c.weight)
                nn.init.uniform_(self.fc2c.bias)
            else:
                nn.init.kaiming_normal_(self.fc1c.weight)
                nn.init.constant_(self.fc1c.bias, 0)
                nn.init.constant_(self.fc2c.weight, 0)
                nn.init.constant_(self.fc2c.bias, 0)

            # self.bn = nn.BatchNorm2d(out_channels)
            # bn_init(self.bn, 1)
        self.attention = attention

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels) if bn else nn.Identity()
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()  # DM
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m, uniform=uniform_init)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        if bn:  # DM
            bn_init(self.bn, 1e-6)

        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset, uniform=uniform_init)

    def forward(self, x, return_attention=False):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            # A = A + self.PA
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                A1 = A[i] + A1 * self.alpha  # DM A1 is C_k in the paper. A[i] is B_k
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))  # DM this is the last orange block in the paper
                y = z + y if y is not None else z
        else:
            A = self.A.cuda(x.get_device())  # * self.mask  # DM: removed multiplication by mask which is not defined
            for i in range(self.num_subset):
                A1 = A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        if return_attention:
            attention = {}

        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            if return_attention:
                attention['spatial'] = se1.detach().squeeze()

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # a2 = se1.unsqueeze(-1)

            if return_attention:
                attention['temporal'] = se1.detach().squeeze()

            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)

            if return_attention:
                attention['channel'] = se2.detach().squeeze()

            # unified attention
            # y = y * self.Attention + y
            # y = y + y * ((a2 + a3) / 2)
            # y = self.bn(y)

        if return_attention:
            return y, attention
        else:
            return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, attention=True, bn=True,
                 temporal_kernel=9, uniform_init=False):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, attention=attention, bn=bn,
                             uniform_init=uniform_init)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride, bn=bn, kernel_size=temporal_kernel,
                             uniform_init=uniform_init)
        self.relu = nn.ReLU(inplace=True)
        # if attention:
        # self.alpha = nn.Parameter(torch.zeros(1))
        # self.beta = nn.Parameter(torch.ones(1))
        # temporal attention
        # self.conv_ta1 = nn.Conv1d(out_channels, out_channels//rt, 9, padding=4)
        # self.bn = nn.BatchNorm2d(out_channels)
        # bn_init(self.bn, 1)
        # self.conv_ta2 = nn.Conv1d(out_channels, 1, 9, padding=4)
        # nn.init.kaiming_normal_(self.conv_ta1.weight)
        # nn.init.constant_(self.conv_ta1.bias, 0)
        # nn.init.constant_(self.conv_ta2.weight, 0)
        # nn.init.constant_(self.conv_ta2.bias, 0)

        # rt = 4
        # self.inter_c = out_channels // rt
        # self.conv_ta1 = nn.Conv2d(out_channels, out_channels // rt, 1)
        # self.conv_ta2 = nn.Conv2d(out_channels, out_channels // rt, 1)
        # nn.init.constant_(self.conv_ta1.weight, 0)
        # nn.init.constant_(self.conv_ta1.bias, 0)
        # nn.init.constant_(self.conv_ta2.weight, 0)
        # nn.init.constant_(self.conv_ta2.bias, 0)
        # s attention
        # num_jpts = A.shape[-1]
        # ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        # pad = (ker_jpt - 1) // 2
        # self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
        # nn.init.constant_(self.conv_sa.weight, 0)
        # nn.init.constant_(self.conv_sa.bias, 0)

        # channel attention
        # rr = 16
        # self.fc1c = nn.Linear(out_channels, out_channels // rr)
        # self.fc2c = nn.Linear(out_channels // rr, out_channels)
        # nn.init.kaiming_normal_(self.fc1c.weight)
        # nn.init.constant_(self.fc1c.bias, 0)
        # nn.init.constant_(self.fc2c.weight, 0)
        # nn.init.constant_(self.fc2c.bias, 0)
        #
        # self.softmax = nn.Softmax(-2)
        # self.sigmoid = nn.Sigmoid()
        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride, bn=bn,
                                     uniform_init=uniform_init)

    def forward(self, x, return_attention=False):
        if self.attention:
            gcn_output = self.gcn1(x, return_attention=return_attention)
            tcn_input = gcn_output[0] if return_attention else gcn_output
            y = self.relu(self.tcn1(tcn_input) + self.residual(x))

            # spatial attention
            # se = y.mean(-2)  # N C V
            # se1 = self.sigmoid(self.conv_sa(se))
            # y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            # se = y.mean(-1)  # N C T
            # # se1 = self.relu(self.bn(self.conv_ta1(se)))
            # se2 = self.sigmoid(self.conv_ta2(se))
            # # y = y * se1.unsqueeze(-1) + y
            # a2 = se2.unsqueeze(-1)

            # se = y  # NCTV
            # N, C, T, V = y.shape
            # se1 = self.conv_ta1(se).permute(0, 2, 1, 3).contiguous().view(N, T, self.inter_c * V)  # NTCV
            # se2 = self.conv_ta2(se).permute(0, 1, 3, 2).contiguous().view(N, self.inter_c * V, T)  # NCVT
            # a2 = self.softmax(torch.matmul(se1, se2) / np.sqrt(se1.size(-1)))  # N T T
            # y = torch.matmul(y.permute(0, 1, 3, 2).contiguous().view(N, C * V, T), a2) \
            #         .view(N, C, V, T).permute(0, 1, 3, 2) * self.alpha + y

            # channel attention
            # se = y.mean(-1).mean(-1)
            # se1 = self.relu(self.fc1c(se))
            # se2 = self.sigmoid(self.fc2c(se1))
            # # y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)
            #
            # y = y * ((a2 + a3) / 2) + y
            # y = self.bn(y)
        else:
            y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))

        if self.attention and return_attention:
            return y, gcn_output[1]
        else:
            return y


class Model(nn.Module):
    def __init__(self, num_class=3, num_point=25, num_person=1, graph=None, in_channels=2,
                 drop_out=0, adaptive=True, attention=True, bn=True, internal_bn=True, with_fc=True, tk=9):
        super(Model, self).__init__()
        assert graph is not None
        self.graph = graph

        A = self.graph.A
        self.num_class = num_class
        self.with_fc = with_fc

        if bn:
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
            bn_init(self.data_bn, 1)
        else:
            self.data_bn = None


        # DM
        self.tgcn_blocks = nn.ModuleList([
            TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive, attention=attention, bn=internal_bn,
                         temporal_kernel=tk),
            TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention, bn=internal_bn, temporal_kernel=tk),
            TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention, bn=internal_bn, temporal_kernel=tk),
            TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention, bn=internal_bn, temporal_kernel=tk),
            TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive, attention=attention, bn=internal_bn,
                         temporal_kernel=tk),
            TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention, bn=internal_bn, temporal_kernel=tk),
            TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention, bn=internal_bn, temporal_kernel=tk),
            TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive, attention=attention, bn=internal_bn,
                         temporal_kernel=tk),
            TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention, bn=internal_bn, temporal_kernel=tk),
            TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention, bn=internal_bn, temporal_kernel=tk)])

        if self.with_fc:
            self.fc = nn.Linear(256, num_class)
            nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

        self.drop_out = nn.Dropout(drop_out) if drop_out else nn.Identity()

    def batch_norm(self, x):
        N, C, T, V, M = x.shape

        if self.data_bn is None:
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        else:
            x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        return x

    def forward(self, x, return_attention_for_layers=None, return_features=False, return_before_avg_pool=False,
                avg=True):
        N, C, T, V, M = x.size()

        x = self.batch_norm(x)  # DM
        attention_layers_list = [] if return_attention_for_layers is None else return_attention_for_layers
        attention_maps = {}

        for li, layer in enumerate(self.tgcn_blocks):
            get_attention = li in attention_layers_list
            output = layer(x, return_attention=get_attention)

            if get_attention:
                x = output[0]
                attention_maps[li] = output[1]
            else:
                x = output

        if return_before_avg_pool:
            if attention_maps:
                return x, attention_maps
            else:
                return x

        if avg:
            # N*M,C,T,V
            c_new = x.size(1)
            x = x.view(N, M, c_new, -1)
            x = x.mean(3).mean(1)
        else:
            assert not self.with_fc

        x = self.drop_out(x)

        if not self.with_fc:
            if attention_maps:
                return x, attention_maps
            else:
                return x

        y = self.fc(x)

        if attention_maps:
            if return_features:
                return y, attention_maps, x
            else:
                return y, attention_maps
        else:
            if return_features:
                return y, x
            else:
                return y
