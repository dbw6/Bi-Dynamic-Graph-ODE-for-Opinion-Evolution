import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq as ode
from utils import *
import random
import math

class GRUObservationCellLogvar(torch.nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super().__init__()
        self.gru_d     = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gru_debug = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        ## prep layer and its initialization
        std            = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep    = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size  = input_size
        self.prep_hidden = prep_hidden

    def forward(self, h, p, X_obs, M_obs, i_obs):
        ## only updating rows that have observations
        p_obs        = p[i_obs]

        mean, logvar = torch.chunk(p_obs, 2, dim=1)
        sigma        = torch.exp(0.5 * logvar)
        error        = (X_obs - mean) / sigma

        ## log normal loss, over all observations
        log_lik_c    = np.log(np.sqrt(2*np.pi))
        losses       = 0.5 * ((torch.pow(error, 2) + logvar + 2*log_lik_c) * M_obs)
        if losses.sum()!=losses.sum():
            import ipdb; ipdb.set_trace()

        ## TODO: try removing X_obs (they are included in error)
        gru_input    = torch.stack([X_obs, mean, logvar, error], dim=2).unsqueeze(2)
        gru_input    = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is (sample x feature x prep_hidden)
        gru_input    = gru_input.permute(2, 0, 1)
        gru_input    = (gru_input * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h, losses


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)

class ODEFunc(nn.Module):  # A kind of ODECell in the view of RNN
    def __init__(self, hidden_size, A, dropout=0.0, no_graph=False, no_control=False):
        super(ODEFunc, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A  # N_node * N_node
        # self.nfe = 0
        self.wt = nn.Linear(hidden_size, hidden_size)
        self.no_graph = no_graph
        self.no_control = no_control

    def forward(self, t, x):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        # self.nfe += 1
        if not self.no_graph:
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                x = torch.sparse.mm(self.A, x)
            else:
                x = torch.mm(self.A, x)
        if not self.no_control:
            x = self.wt(x)
        x = self.dropout_layer(x)
        # x = torch.tanh(x)
        x = F.relu(x)
        # !!!!! Not use relu seems doesn't  matter!!!!!! in theory. Converge faster !!! Better than tanh??
        # x = torch.sigmoid(x)
        return x

class ODEFunc1(nn.Module):  # A kind of ODECell in the view of RNN
    def __init__(self, hidden_size, A, dropout=0.0, no_graph=False, no_control=False):
        super(ODEFunc1, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A  # N_node * N_node
        # self.nfe = 0
        self.wt1 = nn.Linear(hidden_size//2, hidden_size//2)
        self.wt2 = nn.Linear(hidden_size//2, hidden_size//2)
        self.wt3 = nn.Linear(hidden_size//2, hidden_size//2)
        self.wt4 = nn.Linear(hidden_size//2, hidden_size//2)
        self.no_graph = no_graph
        self.no_control = no_control

    def forward(self, t, x):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        y1 = x[:,:(self.hidden_size//2)]
        y2 = x[:,(self.hidden_size//2):]
        # self.nfe += 1
        if not self.no_graph:
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                y1 = torch.sparse.mm(self.A, y1)
                y2 = torch.sparse.mm(self.A, y2)
            else:
                y1 = torch.mm(self.A, y1)
                y2 = torch.mm(self.A, y2)
        if not self.no_control:
            y1_1 = self.wt1(y1)
            y2_1 = self.wt2(y2)
            y1_2 = self.wt3(y1)
            y2_2 = self.wt4(y2)
        y1 = y1_1 + y2_1
        y2 = y1_2 + y2_2
        # x = torch.tanh(x)
        x = torch.cat((y1, y2), dim = 1)
        x = self.dropout_layer(x)
        x = F.relu(x)
        # !!!!! Not use relu seems doesn't  matter!!!!!! in theory. Converge faster !!! Better than tanh??
        # x = torch.sigmoid(x)
        return x

    
class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False): #vt,         :param vt:
        """
        :param odefunc: X' = f(X, t, G, W)
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """

        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        # self.integration_time_vector = vt  # time vector
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self, vt, x):
        integration_time_vector = vt.type_as(x)
        vt = vt.to(x.device)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # return out[-1]
        return out[-1] if self.terminal else out  # 100 * 400 * 10

class ODEBlock1(nn.Module):
    def __init__(self, odefunc, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False): #vt,         :param vt:
        """
        :param odefunc: X' = f(X, t, G, W)
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """

        super(ODEBlock1, self).__init__()
        self.odefunc = odefunc
        # self.integration_time_vector = vt  # time vector
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self, vt, x):
        integration_time_vector = vt.type_as(x)
        vt = vt.to(x.device)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # return out[-1]
        return out[-1] if self.terminal else out  # 100 * 400 * 10

class NDCN(nn.Module):  # myModel
    def __init__(self, input_size, hidden_size, A, num_classes, x0, dropout=0.0,
                 no_embed=False, no_graph=False, no_control=False,
                 rtol=.01, atol=.001, method='dopri5'):
        super(NDCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.A = A  # N_node * N_node
        self.num_classes = num_classes

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.no_embed = no_embed
        self.no_graph = no_graph
        self.no_control = no_control

        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.x0 = x0
        self.mask = (x0 == 0).float()
        self.delta_x = nn.Parameter(torch.empty(x0.shape))
        torch.nn.init.xavier_uniform_(self.delta_x)
        self.preprocess_layer = nn.Sequential(nn.Linear(input_size, num_classes, bias=True), nn.Tanh())
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.neural_dynamic_layer = ODEBlock(
                ODEFunc(hidden_size, A, dropout=dropout, no_graph=no_graph, no_control=no_control),  # OM
                rtol= rtol, atol= atol, method= method)  # t is like  continuous depth
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, num_classes, bias=True), nn.Tanh())

    def forward(self, vt, x=None):
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        if x is None:
            x = self.x0 + self.preprocess_layer(self.delta_x * self.mask)

        if not self.no_embed:
            x = self.input_layer(x)
        hvx = self.neural_dynamic_layer(vt, x)
        output = self.output_layer(hvx)
        return output

class NDCN1(nn.Module):  # myModel
    def __init__(self, input_size, hidden_size, A, num_classes, n, dropout=0.0,
                 no_embed=False, no_graph=False, no_control=False,
                 rtol=.01, atol=.001, method='dopri5'):
        super(NDCN1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.A = A  # N_node * N_node
        self.num_classes = num_classes
        self.n = n

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.no_embed = no_embed
        self.no_graph = no_graph
        self.no_control = no_control

        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.preprocess_layer = nn.Sequential(nn.Linear(input_size, num_classes, bias=True), nn.Tanh()) #nn.Tanh() 
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.neural_dynamic_layer = ODEBlock(
                ODEFunc(hidden_size, A, dropout=dropout, no_graph=no_graph, no_control=no_control),  # OM
                rtol= rtol, atol= atol, method= method)  # t is like  continuous depth
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, num_classes, bias=True), nn.Tanh())
        #self.x0 = nn.Parameter(torch.zeros(n, input_size))
        #self.x0 = nn.Parameter(torch.randn(n, input_size))
        #self.x0 = nn.Parameter(torch.empty(n, input_size))
        #torch.nn.init.xavier_uniform_(self.x0)
        #torch.nn.init.xavier_normal_(self.x0, gain=nn.init.calculate_gain('tanh'))
        # 在 __init__ 方法内部
        total_elements = n * input_size  # n是节点数，input_size是每个节点的维度
        num_minus_ones = int(3 * total_elements / 5)  # -1的数量，按照3:2的比例
        num_ones = total_elements - num_minus_ones    # 1的数量

        # 创建一个包含正确数量的-1和1的数组
        values = [-1] * num_minus_ones + [1] * num_ones
        random.shuffle(values)  # 随机排列

        # 将一维数组转化为二维形状 (n, input_size) 的tensor
        initial_values = torch.tensor(values).view(n, input_size).float()

        # 将initial_values赋值为self.x0的初值
        self.x0 = nn.Parameter(initial_values)

    def forward(self, vt):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        x = self.preprocess_layer(self.x0)
        if not self.no_embed:
            x = self.input_layer(x)
        hvx = self.neural_dynamic_layer(vt, x)
        output = self.output_layer(hvx)
        return output

class NDCN2(nn.Module):  # myModel
    def __init__(self, input_size, hidden_size, A, num_classes, n, dropout=0.0,
                 no_embed=False, no_graph=False, no_control=False,
                 rtol=.01, atol=.001, method='dopri5'):
        super(NDCN2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.A = A  # N_node * N_node
        self.num_classes = num_classes
        self.n = n

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.no_embed = no_embed
        self.no_graph = no_graph
        self.no_control = no_control

        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.preprocess_layer1 = nn.Sequential(nn.Linear(input_size, input_size, bias=True), nn.Sigmoid())
        self.preprocess_layer2 = nn.Sequential(nn.Linear(input_size, input_size, bias=True), nn.Sigmoid())
        self.input_layer1 = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.input_layer2 = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.neural_dynamic_layer = ODEBlock(
                ODEFunc1(hidden_size*2, A, dropout=dropout, no_graph=no_graph, no_control=no_control),# OM
                rtol= rtol, atol= atol, method= method)  # t is like  continuous depth
        self.output_layer1 = nn.Sequential(nn.Linear(hidden_size, num_classes, bias=True), nn.Sigmoid())
        self.output_layer2 = nn.Sequential(nn.Linear(hidden_size, num_classes, bias=True), nn.Sigmoid())
        #self.x0 = nn.Parameter(torch.zeros(n, input_size))
        #self.x0 = nn.Parameter(torch.randn(n, input_size))
        #self.x0 = nn.Parameter(torch.empty(n, input_size))
        #torch.nn.init.xavier_uniform_(self.x0)
        #torch.nn.init.xavier_normal_(self.x0, gain=nn.init.calculate_gain('tanh'))
        # 在 __init__ 方法内部
        total_elements = n * input_size  # n是节点数，input_size是每个节点的维度
        num_minus_ones = int(total_elements  / 2)  # -1的数量，按照3:2的比例
        num_ones = total_elements - num_minus_ones    # 1的数量

        # 创建一个包含正确数量的-1和1的数组
        values = [-1] * num_minus_ones + [1] * num_ones
        random.shuffle(values)  # 随机排列

        # 将一维数组转化为二维形状 (n, input_size) 的tensor
        initial_values = torch.tensor(values).view(n, input_size).float()

        self.y1 = nn.Parameter(torch.relu(initial_values))
        self.y2 = nn.Parameter(torch.relu(-initial_values))
        # 将initial_values赋值为self.x0的初值
    def forward(self, vt):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        y1 = self.preprocess_layer1(self.y1)
        y2 = self.preprocess_layer2(self.y2)
        x0 = y1-y2
        if not self.no_embed:
            y1 = self.input_layer1(y1)
            y2 = self.input_layer2(y2)
        x = torch.cat((y1, y2), dim = 1)
        hvx = self.neural_dynamic_layer(vt, x)
        output1 = self.output_layer1(hvx[:,:, :self.hidden_size])
        output2 = self.output_layer2(hvx[:,:, self.hidden_size:])
        return output1, output2, x0
    
class NDCN3(nn.Module):  # myModel
    def __init__(self, input_size, hidden_size, A, num_classes, n, dropout=0.0,
                 no_embed=False, no_graph=False, no_control=False,
                 rtol=.01, atol=.001, method='dopri5'):
        super(NDCN3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.A = A  # N_node * N_node
        self.num_classes = num_classes
        self.n = n

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.no_embed = no_embed
        self.no_graph = no_graph
        self.no_control = no_control

        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.preprocess_layer = nn.Sequential(nn.Linear(input_size, num_classes, bias=True), nn.Tanh()) #nn.Tanh() 
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.neural_dynamic_layer = ODEBlock(
                ODEFunc(hidden_size, A, dropout=dropout, no_graph=no_graph, no_control=no_control),  # OM
                rtol= rtol, atol= atol, method= method)  # t is like  continuous depth
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, num_classes, bias=True), nn.Tanh())
        #self.x0 = nn.Parameter(torch.zeros(n, input_size))
        #self.x0 = nn.Parameter(torch.randn(n, input_size))
        # self.x0 = nn.Parameter(torch.empty(n, input_size))
        # torch.nn.init.xavier_uniform_(self.x0)
        #torch.nn.init.xavier_normal_(self.x0, gain=nn.init.calculate_gain('tanh'))
        # 在 __init__ 方法内部
        total_elements = n * input_size  # n是节点数，input_size是每个节点的维度
        num_minus_ones = int(total_elements *2/ 3)  # -1的数量，按照3:2的比例
        num_ones = total_elements - num_minus_ones    # 1的数量

        # 创建一个包含正确数量的-1和1的数组
        values = [-1] * num_minus_ones + [1] * num_ones
        random.shuffle(values)  # 随机排列

        # 将一维数组转化为二维形状 (n, input_size) 的tensor
        initial_values = torch.tensor(values).view(n, input_size).float()

        # 将initial_values赋值为self.x0的初值
        self.x0 = nn.Parameter(initial_values)

    def forward(self, vt):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        x0 = self.preprocess_layer(self.x0)
        if not self.no_embed:
            x = self.input_layer(x0)
        hvx = self.neural_dynamic_layer(vt, x)
        output = self.output_layer(hvx)
        return output, x0
    
class GraphConvolution(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, input, propagation_adj):
        support = self.fc(input)
        # CAUTION: Pytorch only supports sparse * dense matrix multiplication
        # CAUTION: Pytorch does not support sparse * sparse matrix multiplication !!!
        # output = torch.sparse.mm(propagation_adj, support)
        output = torch.mm(propagation_adj, support)
        # output = torch.reshape(output, (1, -1)).contiguous()
        return output.view(1, -1)

class NDCN4(nn.Module):  # myModel
    def __init__(self, input_size, hidden_size, A, num_classes, n, dropout=0.0,
                 no_embed=False, no_graph=False, no_control=False,
                 rtol=.01, atol=.001, method='dopri5'):
        super(NDCN4, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.A = A  # N_node * N_node
        self.num_classes = num_classes
        self.n = n

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.no_embed = no_embed
        self.no_graph = no_graph
        self.no_control = no_control

        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.preprocess_layer1 = nn.Sequential(nn.Linear(input_size, input_size, bias=True), nn.ReLU())
        self.preprocess_layer2 = nn.Sequential(nn.Linear(input_size, input_size, bias=True), nn.ReLU())
        self.input_layer1 = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.input_layer2 = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.neural_dynamic_layer = ODEBlock(
                ODEFunc(hidden_size*2, A, dropout=dropout, no_graph=no_graph, no_control=no_control),  # OM
                rtol= rtol, atol= atol, method= method)  # t is like  continuous depth
        self.output_layer1 = nn.Sequential(nn.Linear(hidden_size, num_classes, bias=True), nn.Sigmoid())
        self.output_layer2 = nn.Sequential(nn.Linear(hidden_size, num_classes, bias=True), nn.Sigmoid())
        #self.x0 = nn.Parameter(torch.zeros(n, input_size))
        #self.x0 = nn.Parameter(torch.randn(n, input_size))
        #self.x0 = nn.Parameter(torch.empty(n, input_size))
        #torch.nn.init.xavier_uniform_(self.x0)
        #torch.nn.init.xavier_normal_(self.x0, gain=nn.init.calculate_gain('tanh'))
        # 在 __init__ 方法内部
        total_elements = n * input_size  # n是节点数，input_size是每个节点的维度
        num_minus_ones = int(3 * total_elements / 5)  # -1的数量，按照3:2的比例
        num_ones = total_elements - num_minus_ones    # 1的数量

        # 创建一个包含正确数量的-1和1的数组
        values = [-1] * num_minus_ones + [1] * num_ones
        random.shuffle(values)  # 随机排列

        # 将一维数组转化为二维形状 (n, input_size) 的tensor
        initial_values = torch.tensor(values).view(n, input_size).float()

        # self.y1 = nn.Parameter(torch.relu(initial_values))
        # self.y2 = nn.Parameter(torch.relu(-initial_values))
        self.y1 = torch.relu(initial_values)
        self.y2 = torch.relu(-initial_values)
        # 将initial_values赋值为self.x0的初值
    def forward(self, vt):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        # y1 = self.preprocess_layer1(self.y1)
        # y2 = self.preprocess_layer2(self.y2)
        y1 = self.y1
        y2 = self.y2
        x0 = y1-y2
        if not self.no_embed:
            y1 = self.input_layer1(y1)
            y2 = self.input_layer2(y2)
            x = torch.cat((y1, y2), dim = 1)
        hvx = self.neural_dynamic_layer(vt, x)
        output1 = self.output_layer1(hvx[:,:, :self.hidden_size])
        output2 = self.output_layer2(hvx[:,:, self.hidden_size:])
        return output1, output2, x0
    
class TemporalGCN(nn.Module):
    def __init__(self, input_size_gnn, hidden_size_gnn, input_n_graph, hidden_size_rnn, A, dropout=0.5, rnn_type='lstm'):
        super(TemporalGCN, self).__init__()
        self.input_size_gnn = input_size_gnn
        self.hidden_size_gnn = hidden_size_gnn
        self.input_size_rnn = input_n_graph * hidden_size_gnn
        self.hidden_size_rnn = hidden_size_rnn
        self.output_size = input_n_graph
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A

        self.gc = GraphConvolution(input_size_gnn, hidden_size_gnn)

        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTMCell(self.input_size_rnn, hidden_size_rnn)
        elif rnn_type =='gru':
            self.rnn = nn.GRUCell(self.input_size_rnn, hidden_size_rnn)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNNCell(self.input_size_rnn, hidden_size_rnn)

        self.linear = nn.Linear(hidden_size_rnn, input_n_graph)

    def forward(self, input, future=0):
        outputs = []
        # torch.double  h_t: 1*20  1 sample, 20 hidden in rnn
        h_t = torch.zeros(1, self.hidden_size_rnn, device=input.device)  # torch.zeros(1, self.hidden_size_rnn, dtype=torch.float)
        if self.rnn_type == 'lstm':
            c_t = torch.zeros(1, self.hidden_size_rnn, device=input.device)  # torch.zeros(1, self.hidden_size_rnn, dtype=torch.float)  # dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t = self.dropout_layer(input_t)  # input_t : 400*1
            input_t = self.gc(input_t, self.A)  # input_t 400*2 --> 1*800
            input_t = F.relu(input_t)  # 1*800
            if self.rnn_type == 'lstm':
                h_t, c_t = self.rnn(input_t, (h_t, c_t))  # input_t 1*800  h_t: 1*10
            elif self.rnn_type == 'gru':
                h_t = self.rnn(input_t, h_t )
            elif self.rnn_type == 'rnn':
                h_t = self.rnn(input_t, h_t)

            output = self.linear(h_t).t()  # 400*1
            outputs += [output]
        for i in range(future):  # if we should predict the future
            input_t = self.dropout_layer(output)  # input_t : 400*1
            input_t = self.gc(input_t, self.A)  # input_t 400*20 --> 1*8000
            input_t = F.relu(input_t)

            if self.rnn_type == 'lstm':
                h_t, c_t = self.rnn(input_t, (h_t, c_t))  # input_t 1*8000  h_t: 1*20
            elif self.rnn_type == 'gru':
                h_t = self.rnn(input_t, h_t)
            elif self.rnn_type == 'rnn':
                h_t = self.rnn(input_t, h_t)

            output = self.linear(h_t).t()
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

class NDCN(nn.Module):  # myModel
    def __init__(self, input_size, hidden_size, A, num_classes, n, dropout=0.0,
                 no_embed=False, no_graph=False, no_control=False,
                 rtol=.01, atol=.001, method='dopri5'):
        super(NDCN4, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.A = A  # N_node * N_node
        self.num_classes = num_classes
        self.n = n

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.no_embed = no_embed
        self.no_graph = no_graph
        self.no_control = no_control

        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.preprocess_layer1 = nn.Sequential(nn.Linear(input_size, input_size, bias=True), nn.ReLU())
        self.preprocess_layer2 = nn.Sequential(nn.Linear(input_size, input_size, bias=True), nn.ReLU())
        self.input_layer1 = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.input_layer2 = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.neural_dynamic_layer = ODEBlock(
                ODEFunc(hidden_size*2, A, dropout=dropout, no_graph=no_graph, no_control=no_control),  # OM
                rtol= rtol, atol= atol, method= method)  # t is like  continuous depth
        self.output_layer1 = nn.Sequential(nn.Linear(hidden_size, num_classes, bias=True), nn.Sigmoid())
        self.output_layer2 = nn.Sequential(nn.Linear(hidden_size, num_classes, bias=True), nn.Sigmoid())
        #self.x0 = nn.Parameter(torch.zeros(n, input_size))
        #self.x0 = nn.Parameter(torch.randn(n, input_size))
        #self.x0 = nn.Parameter(torch.empty(n, input_size))
        #torch.nn.init.xavier_uniform_(self.x0)
        #torch.nn.init.xavier_normal_(self.x0, gain=nn.init.calculate_gain('tanh'))
        # 在 __init__ 方法内部
        total_elements = n * input_size  # n是节点数，input_size是每个节点的维度
        num_minus_ones = int(3 * total_elements / 5)  # -1的数量，按照3:2的比例
        num_ones = total_elements - num_minus_ones    # 1的数量

        # 创建一个包含正确数量的-1和1的数组
        values = [-1] * num_minus_ones + [1] * num_ones
        random.shuffle(values)  # 随机排列

        # 将一维数组转化为二维形状 (n, input_size) 的tensor
        initial_values = torch.tensor(values).view(n, input_size).float()

        self.y1 = nn.Parameter(torch.relu(initial_values))
        self.y2 = nn.Parameter(torch.relu(-initial_values))
        self.y1 = torch.relu(initial_values)
        self.y2 = torch.relu(-initial_values)
        # 将initial_values赋值为self.x0的初值
    def forward(self, vt):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        # y1 = self.preprocess_layer1(self.y1)
        # y2 = self.preprocess_layer2(self.y2)
        y1 = self.y1
        y2 = self.y2
        x0 = y1-y2
        if not self.no_embed:
            y1 = self.input_layer1(y1)
            y2 = self.input_layer2(y2)
            x = torch.cat((y1, y2), dim = 1)
        hvx = self.neural_dynamic_layer(vt, x)
        output1 = self.output_layer1(hvx[:,:, :self.hidden_size])
        output2 = self.output_layer2(hvx[:,:, self.hidden_size:])
        return output1, output2, x0
    
class TemporalGCN(nn.Module):
    def __init__(self, input_size_gnn, hidden_size_gnn, input_n_graph, hidden_size_rnn, A, dropout=0.5, rnn_type='lstm'):
        super(TemporalGCN, self).__init__()
        self.input_size_gnn = input_size_gnn
        self.hidden_size_gnn = hidden_size_gnn
        self.input_size_rnn = input_n_graph * hidden_size_gnn
        self.hidden_size_rnn = hidden_size_rnn
        self.output_size = input_n_graph
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A

        self.gc = GraphConvolution(input_size_gnn, hidden_size_gnn)

        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTMCell(self.input_size_rnn, hidden_size_rnn)
        elif rnn_type =='gru':
            self.rnn = nn.GRUCell(self.input_size_rnn, hidden_size_rnn)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNNCell(self.input_size_rnn, hidden_size_rnn)

        self.linear = nn.Linear(hidden_size_rnn, input_n_graph)

    def forward(self, input, future=0):
        outputs = []
        # torch.double  h_t: 1*20  1 sample, 20 hidden in rnn
        h_t = torch.zeros(1, self.hidden_size_rnn, device=input.device)  # torch.zeros(1, self.hidden_size_rnn, dtype=torch.float)
        if self.rnn_type == 'lstm':
            c_t = torch.zeros(1, self.hidden_size_rnn, device=input.device)  # torch.zeros(1, self.hidden_size_rnn, dtype=torch.float)  # dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t = self.dropout_layer(input_t)  # input_t : 400*1
            input_t = self.gc(input_t, self.A)  # input_t 400*2 --> 1*800
            input_t = F.relu(input_t)  # 1*800
            if self.rnn_type == 'lstm':
                h_t, c_t = self.rnn(input_t, (h_t, c_t))  # input_t 1*800  h_t: 1*10
            elif self.rnn_type == 'gru':
                h_t = self.rnn(input_t, h_t )
            elif self.rnn_type == 'rnn':
                h_t = self.rnn(input_t, h_t)

            output = self.linear(h_t).t()  # 400*1
            outputs += [output]
        for i in range(future):  # if we should predict the future
            input_t = self.dropout_layer(output)  # input_t : 400*1
            input_t = self.gc(input_t, self.A)  # input_t 400*20 --> 1*8000
            input_t = F.relu(input_t)

            if self.rnn_type == 'lstm':
                h_t, c_t = self.rnn(input_t, (h_t, c_t))  # input_t 1*8000  h_t: 1*20
            elif self.rnn_type == 'gru':
                h_t = self.rnn(input_t, h_t)
            elif self.rnn_type == 'rnn':
                h_t = self.rnn(input_t, h_t)

            output = self.linear(h_t).t()
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
# class ODEBlock(nn.Module):
#     def __init__(self, odefunc, vt, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False):
#         """
#         :param odefunc: X' = f(X, t, G, W)
#         :param vt:
#         :param rtol: optional float64 Tensor specifying an upper bound on relative error,
#             per element of `y`.
#         :param atol: optional float64 Tensor specifying an upper bound on absolute error,
#             per element of `y`.
#         :param method:
#             'explicit_adams': AdamsBashforth,
#             'fixed_adams': AdamsBashforthMoulton,
#             'adams': VariableCoefficientAdamsBashforth,
#             'tsit5': Tsit5Solver,
#             'dopri5': Dopri5Solver,
#             'euler': Euler,
#             'midpoint': Midpoint,
#             'rk4': RK4,
#         """
#
#         super(ODEBlock, self).__init__()
#         self.odefunc = odefunc
#         self.integration_time_vector = vt  # time vector
#         self.rtol = rtol
#         self.atol = atol
#         self.method = method
#         self.adjoint = adjoint
#         self.terminal = terminal
#
#     def forward(self, x):
#         self.integration_time_vector = self.integration_time_vector.type_as(x)
#         if self.adjoint:
#             out = ode.odeint_adjoint(self.odefunc, x, self.integration_time_vector,
#                                      rtol=self.rtol, atol=self.atol, method=self.method)
#         else:
#             out = ode.odeint(self.odefunc, x, self.integration_time_vector,
#                              rtol=self.rtol, atol=self.atol, method=self.method)
#         # return out[-1]
#         return out[-1] if self.terminal else out  # 100 * 400 * 10

# TO BE DELETED!
# class GraphOperator(nn.Module):
#     def __init__(self,  alpha=True):
#         super(GraphOperator, self).__init__()
#         if alpha:
#             self.alpha = nn.Parameter(torch.FloatTensor([0.5]))
#         else:
#             self.register_parameter('alpha', None)
#
#     def forward(self, A, x):  # How to use t?
#         """
#         :param t:  end time tick, if t is not used, it is an autonomous system
#         :param x:  initial value   N_node * N_dim   400 * hidden_size
#         :return:
#         """
#         A_prime = self.alpha * A + (1.0-self.alpha) * torch.eye(A.shape[0]).cuda()
#         out_degree = A_prime.sum(1)
#         # in_degree = A_prime.sum(0)
#
#         out_degree_sqrt_inv = torch.matrix_power(torch.diag(out_degree), -1)
#         out_degree_sqrt_inv[torch.isinf(out_degree_sqrt_inv)] = 0.0
#         # int_degree_sqrt_inv = torch.matrix_power(torch.diag(in_degree), -0.5)
#         # int_degree_sqrt_inv[torch.isinf(int_degree_sqrt_inv)] = 0.0
#         mx_operator = torch.mm(out_degree_sqrt_inv, A_prime)
#         x = torch.mm(mx_operator, x)
#         return x
#
#
# class ODEFunc_A(nn.Module):
#     def __init__(self, hidden_size, A, dropout=0.0, no_graph=False, no_control=False):
#         super(ODEFunc_A, self).__init__()
#         self.hidden_size = hidden_size
#         self.dropout = dropout
#         self.dropout_layer = nn.Dropout(dropout)
#         self.A = A  # N_node * N_node
#         # self.nfe = 0
#         self.wt = nn.Linear(hidden_size, hidden_size)
#         self.no_graph = no_graph
#         self.no_control = no_control
#         self.GraphOperator = GraphOperator(alpha=True)
#
#     def forward(self, t, x):  # How to use t?
#         """
#         :param t:  end time tick, if t is not used, it is an autonomous system
#         :param x:  initial value   N_node * N_dim   400 * hidden_size
#         :return:
#         """
#
#         x = self.GraphOperator.forward(self.A, x)
#         if not self.no_control:
#             x = self.wt(x)
#         x = self.dropout_layer(x)
#         x = F.relu(x)
#         return x
