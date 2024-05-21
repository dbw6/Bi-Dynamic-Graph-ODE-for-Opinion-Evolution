import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import networkx as nx
from networkx.algorithms import community
import matplotlib.cm as cm
import matplotlib.dates as mdates
from sklearn.metrics import f1_score


def custom_sign(tensor):
    return torch.where(tensor <= 0, torch.tensor(-1, dtype=tensor.dtype), torch.tensor(1, dtype=tensor.dtype))

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def draw_loss(abs_errors, rel_errors, niters, test_freq):
    """
    绘制损失随训练次数变化的图表。
    
    参数：
    abs_errors (list): 绝对误差列表。
    rel_errors (list): 相对误差列表。
    niters (int): 总训练次数。
    test_freq (int): 测试频率。
    """
    # Create a line plot for absolute errors
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, niters + 1), abs_errors, label='Absolute Error')
    plt.xlabel('Iterations')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error vs. Iterations')
    plt.legend()
    plt.grid()
    plt.show()

    # Create a line plot for relative errors
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, niters + 1), rel_errors, label='Relative Error')
    plt.xlabel('Iterations')
    plt.ylabel('Relative Error')
    plt.title('Relative Error vs. Iterations')
    plt.legend()
    plt.grid()
    plt.show()





import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
def just_calculate_infection_ratio(A, true_y):
    num_users, num_days = true_y.shape
    negative_infection_ratio_list = []
    positive_infection_ratio_list = []

    for day in range(num_days):  # 从第一天开始计算
        # 在每天开始时重置 already_infected_users
        already_infected_users = set()

        # 找到当天观点为负的用户
        negative_users = torch.where(true_y[:, day] == -1)[0]
        # 找到当天观点为正的用户
        positive_users = torch.where(true_y[:, day] == 1)[0]

        # 找到与这些观点为负的用户相连的所有其他用户
        possible_negative_infected_users = torch.where(torch.sum(A[negative_users, :], dim=0) > 0)[0]
        # 找到与这些观点为正的用户相连的所有其他用户
        possible_positive_infected_users = torch.where(torch.sum(A[positive_users, :], dim=0) > 0)[0]

        # 初始化 real_negative_infected_users 和 real_positive_infected_users
        real_negative_infected_users = torch.tensor([], dtype=torch.long)
        real_positive_infected_users = torch.tensor([], dtype=torch.long)

        # 在可能负感染者中找到那些在前一天观点为0在当天转变为-1的用户
        if day > 0:
            neutral_users_previous_day = torch.where(true_y[:, day - 1] != -1)[0]
            possible_negative_infected_users = torch.tensor(list(
                set(neutral_users_previous_day.tolist()).intersection(set(possible_negative_infected_users.tolist()))))
            real_negative_infected_users = torch.tensor(list(
                set(possible_negative_infected_users.tolist()).intersection(
                    set(torch.where(true_y[:, day] == -1)[0].tolist()))))
            real_negative_infected_users = torch.tensor(
                list(set(real_negative_infected_users.tolist()).difference(already_infected_users)))  # 排除已经被感染的用户
            already_infected_users.update(real_negative_infected_users.tolist())  # 更新已经被感染的用户列表

        # 在可能正感染者中找到那些在前一天观点为0在当天转变为1的用户
        if day > 0:
            neutral_users_previous_day = torch.where(true_y[:, day - 1] != 1)[0]
            possible_positive_infected_users = torch.tensor(list(
                set(neutral_users_previous_day.tolist()).intersection(set(possible_positive_infected_users.tolist()))))
            real_positive_infected_users = torch.tensor(list(
                set(possible_positive_infected_users.tolist()).intersection(
                    set(torch.where(true_y[:, day] == 1)[0].tolist()))))
            real_positive_infected_users = torch.tensor(
                list(set(real_positive_infected_users.tolist()).difference(already_infected_users)))  # 排除已经被感染的用户
            already_infected_users.update(real_positive_infected_users.tolist())  # 更新已经被感染的用户列表

        # 计算每天的‘真实负感染者’数量和‘真实正感染者’数量
        if (len(possible_negative_infected_users) != 0):
            negative_infection_ratio = len(real_negative_infected_users) / len(possible_negative_infected_users)
        else:
            negative_infection_ratio = 0

        if (len(possible_positive_infected_users) != 0):
            positive_infection_ratio = len(real_positive_infected_users) / len(possible_positive_infected_users)
        else:
            positive_infection_ratio = 0

        negative_infection_ratio_list.append(negative_infection_ratio)
        positive_infection_ratio_list.append(positive_infection_ratio)

    print("Negative Infection Ratio List:", negative_infection_ratio_list)
    print("Positive Infection Ratio List:", positive_infection_ratio_list)

    return negative_infection_ratio_list, positive_infection_ratio_list
def caculate_combine_Pbar(A,B,true_y,pred_y):
    t_neg, t_pos = just_calculate_infection_ratio(A, true_y)
    p_neg, p_pos = just_calculate_infection_ratio(B, pred_y)

    labels = ['NEG True', 'POS True', 'NEG Pred', 'POS Pred']
    plot_mean_attitude_changes_and_p_value(
        t_neg, t_pos, p_neg, p_pos, labels,
        xlabel='Group', ylabel='Average Number of People', color_scheme=2, show=True
    )
# Updated function to only plot the bar chart with the average values and P-value


import seaborn as sns


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def plot_mean_attitude_changes_and_p_value(t_neg, t_pos, p_neg, p_pos, labels, xlabel, ylabel, color_scheme, show):
    # 计算每个数组的平均值
    # 排除每个数组的第一个元素
    t_neg = np.array(t_neg[2:])
    t_pos = np.array(t_pos[2:])
    p_neg = np.array(p_neg[2:])
    p_pos = np.array(p_pos[2:])

    # 计算每个数组的平均值
    t_neg_mean = np.mean(t_neg)
    t_pos_mean = np.mean(t_pos)
    p_neg_mean = np.mean(p_neg)
    p_pos_mean = np.mean(p_pos)

    means = [t_neg_mean, t_pos_mean, p_neg_mean,p_pos_mean]

    # 使用元素级除法计算比率数组，并避免除以零
    ture_ratio = t_pos / (t_neg + 1e-10)
    pred_ratio = p_pos / (p_neg + 1e-10)

    # 计算两个比率数组之间的P值
    _, p_value = ttest_ind(ture_ratio, pred_ratio, equal_var=False)

    # 绘制柱状图
    colors = ['green', 'orange', 'green', 'orange'] if color_scheme == 2 else ['gray'] * 4
    plt.figure(figsize=(10, 6))  # 设置图的尺寸
    plt.bar(labels, means, color=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Average Number of People')

    # 在图上标注P值
    plt.text(0.5, max(means) * 0.95, f'P-value: {p_value:.2e}', horizontalalignment='center', fontsize=12)

    if show:
        plt.show()

    return ture_ratio, pred_ratio, p_value




def plot_attitude_changes(dates, attitude_changes_true, attitude_changes_pred, label1, label2,
                          xlabel, ylabel, color_scheme, show=True):
    # 默认颜色方案
    colors = sns.color_palette("hsv", 2)  # 或选择任何其他默认配色方案
    color_true = colors[0]
    color_pred = colors[1]

    if color_scheme == 1:
        colors = sns.color_palette("vlag", 41)  # 蓝红配色
        color_true = colors[0]
        color_pred = colors[-1]
    elif color_scheme == 2:
        colors = sns.color_palette(["#ff7f0e", "#2ca02c"])  # 橙绿配色
        color_true = colors[0]
        color_pred = colors[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x=dates, y=attitude_changes_true, marker='o', markersize=10, linestyle='-', linewidth=7,
                 color=colors[0], label=label1, ax=ax)
    sns.lineplot(x=dates, y=attitude_changes_pred, marker='s', markersize=10, linestyle='-', linewidth=7,
                 color=colors[-1], label=label2, ax=ax)

    # Increase the bottom margin to make space for the title below the plot
    plt.subplots_adjust(bottom=0.1)

    # Set xlabel and ylabel with increased font size
    ax.set_xlabel(f'{xlabel}', fontsize=30)
    ax.set_ylabel(f'{ylabel}', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)

    # Set legend with increased font size
    ax.legend(fontsize=30)

    plt.tight_layout()

    # Show the plot if required
    if show:
        plt.show()
# Example usage:
# plot_attitude_changes(dates, attitude_changes_true, attitude_changes_pred, 'Label 1', 'Label 2')


def plot_attitude_changes_no_date(dates, attitude_changes_true, attitude_changes_pred, title = 'Attitude Changes Over Time', show = False, group_num = 0):
    line1, = plt.plot(dates, attitude_changes_true, marker='o', linestyle='-', color='b')
    line2, = plt.plot(dates, attitude_changes_pred, marker='o', linestyle='-', color='r')

    plt.xlabel('Time')
    plt.ylabel('Number of Users')
    plt.title(title)
    plt.grid(True)
    if group_num != 0:
        plt.text(0.05, 0.95, f"Group: {group_num}", transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white"))
    if show:
        plt.show()
    return [line1, line2]

def plot_attitude_changes_with_seaborn(dates, attitude_changes_true, attitude_changes_pred, label1, label2, title='Attitude Changes Over Time', show=False, group_num=0):
    # 设置 seaborn 的风格
    sns.set_theme(style="whitegrid")
    
    # 创建一个图形实例
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制真实值和预测值
    line1 = sns.lineplot(x=dates, y=attitude_changes_true, marker='o', linestyle='-', color='b', label=label1, ax=ax)
    line2 = sns.lineplot(x=dates, y=attitude_changes_pred, marker='o', linestyle='-', color='r', label=label2, ax=ax)
    
    # 设置标题、坐标轴标签等
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)
    
    if group_num != 0:
        ax.text(0.05, 0.95, f"Group: {group_num}", transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
    
    # 显示图形
    if show:
        plt.show()
    
    return [line1, line2]

def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

def draw_f1(iterations, f1_1_values, f1_2_values):
    plt.figure()
    plt.plot(iterations, f1_1_values, label='test1 F1 Score')
    plt.plot(iterations, f1_2_values, label='test2 F1 Score')
    plt.xlabel('Iterations')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Model F1 Score Over Iterations')
    plt.savefig('f1_curve.png')  # 保存绘图结果
    plt.show()

def draw_loss_and_save(pred_y, true_y, id_train, criterion, filename, group_num = 0):
    num_x = pred_y.shape[1]
    losses = []

    for i in range(num_x):
        pred = pred_y[:,i]
        true = true_y[:,i]
        mask = (true != 0.0)
        loss = (criterion(pred, true, reduction='none') * mask).sum().float()/mask.sum().float()
        losses.append(loss.item())

    plt.figure()

    # Extract the losses corresponding to id_train and plot them with a blue line and triangles
    train_losses = [losses[i] for i in id_train]
    plt.plot(id_train, train_losses, color='blue', marker='^', linestyle='-')

    # Extract the losses NOT corresponding to id_train and plot them with a gray line and squares
    non_train_ids = [i for i in range(num_x) if i not in id_train]
    non_train_losses = [losses[i] for i in non_train_ids]
    plt.plot(non_train_ids, non_train_losses, color='gray', marker='s', linestyle='--')

    # Scatter plot for all points; triangles for id_train and squares for others
    for i, loss in enumerate(losses):
        if i in id_train:
            plt.scatter(i, loss, marker='^', color='blue')  # Train points in blue
        else:
            plt.scatter(i, loss, marker='s', color='gray')  # Non-train points in gray

    if group_num != 0:
        plt.text(0.05, 0.95, f"Group: {group_num}", transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white"))
    
    plt.xlabel('id_train')
    plt.ylabel('loss')
    plt.ylim([0, 1.5])
    plt.title('loss')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def draw_combined_and_save(pred_y, true_y, changed, id_train, criterion, attitude_true, attitude_pred, filename, group_num = 0):
    num_x = pred_y.shape[1]
    losses = []
    
    for i in range(num_x):
        pred = pred_y[:,i]
        true = true_y[:,i]
        change = changed[:,i]
        loss = (criterion(pred, true, reduction='none') * change).sum() / change.sum()
        losses.append(loss.item())

    fig, ax1 = plt.subplots()

    # Plot loss on the left y-axis
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(losses, color='gray', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    for i, loss in enumerate(losses):
        marker = 's'
        if i in id_train:
            marker = '^'
        ax1.scatter(i, loss, color='tab:red', marker=marker)

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Attitude', color='tab:blue')
    ax2.plot(attitude_true, marker='o', linestyle='-', color='b')
    ax2.plot(attitude_pred, marker='o', linestyle='-', color='r')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.title('Combined Loss and Attitude')
    if group_num != 0:
        plt.text(0.05, 0.95, f"Epoch: {group_num}", transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white"))
    plt.savefig(filename)
    plt.close()

def visualize(N, x0, xt, figname, title ='Dynamics in Complex Network', dir='png_learn_dynamics', zmin=None, zmax=None):
    """
    :param N:   N**2 is the number of nodes, N is the pixel of grid
    :param x0:  initial condition
    :param xt:  states at time t to plot
    :param figname:  figname , numbered
    :param title: title in figure
    :param dir: dir to save
    :param zmin: ax.set_zlim(zmin, zmax)
    :param zmax: ax.set_zlim(zmin, zmax)
    :return:
    """
    if zmin is None:
        zmin = x0.min()
    if zmax is None:
        zmax = x0.max()
    fig = plt.figure()  # figsize=(12, 4), facecolor='white'
    fig.tight_layout()
    x0 = x0.detach()
    xt = xt.detach()
    ax = fig.add_subplot(111, projection='3d')
    ax.cla()
    # ax.set_title(title)
    X = np.arange(0, N)
    Y = np.arange(0, N)
    X, Y = np.meshgrid(X, Y)  # X, Y, Z : 20 * 20
    # R = np.sqrt(X ** 2 + Y ** 2)
    # Z = np.sin(R)
    # fig.set_xlabel('t')
    # ax_traj.set_ylabel('x,y')
    # ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
    # ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
    # ax_traj.set_xlim(t.min(), t.max())
    # ax_traj.set_ylim(-2, 5)
    # ax.pcolormesh(xt.view(N,N), cmap=plt.get_cmap('hot'))
    surf = ax.plot_surface(X, Y, xt.numpy().reshape((N, N)), cmap='rainbow',
                           linewidth=0, antialiased=False, vmin=zmin, vmax=zmax)
    ax.set_zlim(zmin, zmax)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    fig.savefig(dir+'/'+figname+".png", transparent=True)
    fig.savefig(dir+'/'+figname + ".pdf", transparent=True)

    # plt.draw()
    plt.pause(0.001)
    plt.close(fig)


def visualize_graph_matrix(G, title, dir=r'figure/network'):
    A = nx.to_numpy_array(G)
    fig = plt.figure()  # figsize=(12, 4), facecolor='white'
    fig.tight_layout()
    plt.imshow(A, cmap='Greys')  # ''YlGn')
    # plt.pcolormesh(A)
    plt.show()

    fig.savefig(dir + '/' + title + ".png", transparent=True)
    fig.savefig(dir + '/' + title + ".pdf", transparent=True)


def zipf_smoothing(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    """
    A_prime = A + np.eye(A.shape[0])
    out_degree = np.array(A_prime.sum(1), dtype=np.float32)
    int_degree = np.array(A_prime.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A_prime @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def normalized_plus(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D ^-1/2 * ( A + I ) * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ (A + np.eye(A.shape[0])) @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def normalized_laplacian(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 * ( D - A ) * D^-1/2 = I - D^-1/2 * ( A ) * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float64)
    int_degree = np.array(A.sum(0), dtype=np.float64)
    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0.0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0.0))
    mx_operator = np.eye(A.shape[0]) - np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def normalized_adj(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 *  A   * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def grid_8_neighbor_graph(N):
    """
    Build discrete grid graph, each node has 8 neighbors
    :param n:  sqrt of the number of nodes
    :return:  A, the adjacency matrix
    """
    N = int(N)
    n = int(N ** 2)
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    A = torch.zeros(n, n)
    for x in range(N):
        for y in range(N):
            index = x * N + y
            for i in range(len(dx)):
                newx = x + dx[i]
                newy = y + dy[i]
                if N > newx >= 0 and N > newy >= 0:
                    index2 = newx * N + newy
                    A[index, index2] = 1
    return A.float()


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()
        self.val = None
        self.avg = 0

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_batch(true_y, t, data_size, batch_time, batch_size, device):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    # s: 20
    batch_y0 = true_y[s]  # (M, D) 500*1*2
    batch_y0 = batch_y0.squeeze() # 500 * 2
    batch_t = t[:batch_time]  # (T) 19
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)
    # (T, M, D) 19*500*1*2   from s and its following batch_time sample
    batch_y = batch_y.squeeze()  # 19 * 500 * 2
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def torch_sensor_to_torch_sparse_tensor(mx):
    """ Convert a torch.tensor to a torch sparse tensor.
    :param torch tensor mx
    :return: torch.sparse
    """
    index = mx.nonzero().t()
    value = mx.masked_select(mx != 0)
    shape = mx.shape
    return torch.sparse.FloatTensor(index, value, shape)


def test():
    a = torch.tensor([[2,0,3], [0,1,-1]]).float()
    print(a)
    b = torch_sensor_to_torch_sparse_tensor(a)
    print(b.to_dense())
    print(b)


def generate_node_mapping(G, type=None):
    """
    :param G:
    :param type:
    :return:
    """
    if type == 'degree':
        s = sorted(G.degree, key=lambda x: x[1], reverse=True)
        new_map = {s[i][0]: i for i in range(len(s))}
    elif type == 'community':
        cs = list(community.greedy_modularity_communities(G))
        l = []
        for c in cs:
            l += list(c)
        new_map = {l[i]:i for i in range(len(l))}
    else:
        new_map = None

    return new_map


def networkx_reorder_nodes(G, type=None):
    """
    :param G:  networkX only adjacency matrix without attrs
    :param nodes_map:  nodes mapping dictionary
    :return:
    """
    nodes_map = generate_node_mapping(G, type)
    if nodes_map is None:
        return G
    C = nx.to_scipy_sparse_matrix(G, format='coo')
    new_row = np.array([nodes_map[x] for x in C.row], dtype=np.int32)
    new_col = np.array([nodes_map[x] for x in C.col], dtype=np.int32)
    new_C = sp.coo_matrix((C.data, (new_row, new_col)), shape=C.shape)
    new_G = nx.from_scipy_sparse_matrix(new_C)
    return new_G


class CustomLossWithDerivative(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CustomLossWithDerivative, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        loss_mse = torch.mean((input - target)**2)  # 均方误差部分

        # 计算导数并计算绝对值
        gradient_input = torch.autograd.grad(outputs=loss_mse, inputs=input, grad_outputs=torch.ones_like(loss_mse), create_graph=True)[0]
        gradient_absolute = torch.abs(gradient_input)

        # 绝对值导数的损失部分
        loss_derivative = torch.mean(gradient_absolute)

        # 最终损失为均方误差部分和绝对值导数部分的加权和
        final_loss = self.alpha * loss_mse + self.beta * loss_derivative
        return final_loss


def test_graph_generator():
    n = 400
    m = 5
    seed = 0
    # G = nx.barabasi_albert_graph(n, m, seed)
    G = nx.random_partition_graph([100, 100, 200], .25, .01)
    sizes = [10, 90, 300]
    probs = [[0.25, 0.05, 0.02],
             [0.05, 0.35, 0.07],
             [0.02, 0.07, 0.40]]
    G = nx.stochastic_block_model(sizes, probs, seed=0)

    G = nx.newman_watts_strogatz_graph(400, 5, 0.5)

    A = nx.to_numpy_array(G)
    print(A)
    plt.pcolormesh(A)
    plt.show()

    s = sorted(G.degree, key=lambda x: x[1], reverse=True)
    # newmap = {s[i][0]:i for i in range(len(s))}
    # H= nx.relabel_nodes(G,newmap)
    # newmap = generate_node_mapping(G, type='community')
    # H = networkX_reorder_nodes(G, newmap)
    H = networkx_reorder_nodes(G, 'community')

    # B = nx.to_numpy_array(H)
    # # plt.pcolormesh(B)
    # plt.imshow(B)
    # plt.show()

    visualize_graph_matrix(H)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total {:d} Trainable {:d}'.format(total_num, trainable_num))
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    test_graph_generator()
