import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine, cityblock, euclidean
from scipy.stats import pearsonr
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch


@torch.no_grad()
def sampling_idx_individual_dst(class_num_list, idx_info, device):
    # Selecting src & dst nodes
    max_num, n_cls = max(class_num_list), len(class_num_list)
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
    new_class_num_list = torch.Tensor(class_num_list).to(device)

    # Compute # of source nodes
    sampling_src_idx = [cls_idx[torch.randint(len(cls_idx), (int(samp_num.item()),))]
                        for cls_idx, samp_num in zip(idx_info, sampling_list)]
    sampling_src_idx = torch.cat(sampling_src_idx)

    # Generate corresponding destination nodes
    prob = torch.log(new_class_num_list.float()) / new_class_num_list.float()
    prob = prob.repeat_interleave(new_class_num_list.long())
    temp_idx_info = torch.cat(idx_info)
    dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)

    sampling_dst_idx = temp_idx_info[dst_idx]

    # Sorting src idx with corresponding dst idx
    sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
    sampling_dst_idx = sampling_dst_idx[sorted_idx]

    return sampling_src_idx, sampling_dst_idx


def saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam):
    new_src = x[sampling_src_idx.to(x.device), :].clone()
    new_dst = x[sampling_dst_idx.to(x.device), :].clone()
    lam = lam.to(x.device)

    mixed_node = lam * new_src + (1 - lam) * new_dst
    new_x = torch.cat([x, mixed_node], dim=0)
    return new_x


@torch.no_grad()
def duplicate_neighbor(total_node, edge_index, sampling_src_idx):
    device = edge_index.device

    # Assign node index for augmented nodes
    row, col = edge_index[0], edge_index[1]
    row, sort_idx = torch.sort(row)
    col = col[sort_idx]
    degree = scatter_add(torch.ones_like(row), row)
    new_row = (torch.arange(len(sampling_src_idx)).to(device) + total_node).repeat_interleave(degree[sampling_src_idx])
    temp = scatter_add(torch.ones_like(sampling_src_idx), sampling_src_idx).to(device)

    # Duplicate the edges of source nodes
    node_mask = torch.zeros(total_node, dtype=torch.bool)
    unique_src = torch.unique(sampling_src_idx)
    node_mask[unique_src] = True
    node_mask = node_mask.to(row.device)
    row_mask = node_mask[row]
    edge_mask = col[row_mask]
    b_idx = torch.arange(len(unique_src)).to(device).repeat_interleave(degree[unique_src])
    edge_dense, _ = to_dense_batch(edge_mask, b_idx, fill_value=-1)
    if len(temp[temp != 0]) != edge_dense.shape[0]:
        cut_num = len(temp[temp != 0]) - edge_dense.shape[0]
        cut_temp = temp[temp != 0][:-cut_num]
    else:
        cut_temp = temp[temp != 0]
    edge_dense = edge_dense.repeat_interleave(cut_temp, dim=0)
    new_col = edge_dense[edge_dense != -1]
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index


@torch.no_grad()
def neighbor_sampling(edge_index, train_edge_mask, sampling_dst_idx):
    """
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        edge_index:         边的索引，尺寸为[2, # of edges]
        train_edge_mask:    训练集边的掩码，尺寸为[# of edges]
        sampling_dst_idx:   采样获得的目标结点的索引，尺寸为 [# of augmented nodes]
    Output:
        new_edge_index:     原始边的索引，将采样获得的目标结点的边的mask改为True，加入训练集
    """
    sampling_dst_idx = torch.tensor(sampling_dst_idx)
    device = edge_index.device
    sampling_dst_idx = sampling_dst_idx.to(device)
    mask = torch.where(torch.isin(edge_index[0], sampling_dst_idx))
    train_edge_mask[mask] = True
    new_edge_index = edge_index[:, train_edge_mask]

    return new_edge_index

def cout_KDE(x, class_num_list, unlabeled_idx, data_train_mask, y):
    x_temp = x.clone()
    # 数据标准化
    scaler = StandardScaler()
    x_temp = x_temp.to("cpu")
    y = y.to("cpu")
    data_train_mask = data_train_mask.to("cpu")
    x_temp = scaler.fit_transform(x_temp)
    train_features = x_temp[data_train_mask]
    unlabeled_features = x_temp[unlabeled_idx]
    class_labeled_node = []
    for i in range(len(class_num_list)):
        class_i_labeled_node = train_features[y[data_train_mask] == i]
        class_labeled_node.append(class_i_labeled_node)

    kde_classes = []
    for i in range(len(class_num_list)):
        print("len(class_labeled_node[i]):", len(class_labeled_node[i]))
        # 分别为每个类别训练KDE
        kde_class = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(class_labeled_node[i])
        kde_classes.append(kde_class)

    prob_classes = []
    for i in range(len(class_num_list)):
        log_dens_class = kde_classes[i].score_samples(unlabeled_features)
        prob_class = np.exp(log_dens_class)
        prob_classes.append(prob_class)

    return prob_classes

def get_similar(x, x_center, prob, a):
    similarity_all = []
    for x_temp in x:
        # 余弦值
        similarity = 1 - cosine(x_center.cpu().numpy(), x_temp.cpu().numpy())
        # 曼哈顿距离
        # similarity = cityblock(x_center.cpu().numpy(), x_temp.cpu().numpy())
        # 欧氏距离
        # similarity = euclidean(x_center.cpu().numpy(), x_temp.cpu().numpy())
        # similarity = np.array(similarity)

        # 计算最小值和最大值
        # min_value = np.min(similarity)
        # max_value = np.max(similarity)
        # 应用归一化公式
        # similarity = (similarity - min_value) / (max_value - min_value)
        # 皮尔逊系数
        # similarity, _ = pearsonr(x_center.cpu().numpy(), x_temp.cpu().numpy())
        similarity_all.append(similarity)

    similarity_all = torch.tensor(similarity_all).to(prob.device)
    for i in range(len(similarity_all)):
        if(prob[i] > a):
            prob[i] = prob[i]
        else:
            prob[i] = similarity_all[i] + prob[i]
    prob = prob + similarity_all
    # prob = similarity_all
    return prob
@torch.no_grad()
def graph_augment(x, class_num_list, prev_out_local, prev_last, unlabeled_idx, data_train_mask, y, a=0.8, tau=2, max_flag=False,):
    '''
    Args:
        x: 图中节点的特征矩阵
        class_num_list: 每个类别节点数量的tensor
        prev_out_local: 未标记节点上一轮GNN结果得到的节点嵌入
        prev_last: 全部节点上一轮GNN结果得到的节点嵌入
        unlabeled_idx: 未标记节点的索引
        data_train_mask: 训练集掩码
        y: 图中节点的标签矩阵
    '''
    y_temp = y.clone()
    # 计算值为 True 的元素个数
    data_train_mask_temp = data_train_mask.clone()
    max_num, n_cls = max(class_num_list), len(class_num_list)
    if not max_flag:  # mean
        max_num = int(sum(class_num_list) / n_cls)
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
    unlabled_idx_temp = unlabeled_idx
    unlabled_idx_temp = torch.tensor(unlabled_idx_temp)
    prev_out_local_prob = F.softmax(prev_out_local / tau, dim=1)

    src_idx_all = []
    dst_idx_all = []
    need_nodes_num = 0
    for cls_idx, num in enumerate(sampling_list):
        num = int(num.item())
        if num <= 0:
            continue
        need_nodes_num += num
    if need_nodes_num > unlabled_idx_temp.numel():
        for cls_idx, num in enumerate(sampling_list):
            num = int(num.item())
            if num <= 0:
                continue
            sampling_list[cls_idx] = int((num / need_nodes_num) * unlabled_idx_temp.numel() * 0.7)
    for cls_idx, num in enumerate(sampling_list):
        #计算训练集少样本的中心（从节点的嵌入空间中计算）
        x_temp = prev_last[data_train_mask].clone()
        y_get_index_temp = y[data_train_mask].clone()
        cls_indices = torch.nonzero(y_get_index_temp == cls_idx, as_tuple=True)
        cls_indices = cls_indices[0].tolist()
        x_all = x_temp[cls_indices]
        # x_center少样本中心
        x_center = x_all.mean(dim=0)
        # 获取未标记节点的嵌入
        x_unlabeled = prev_last[unlabled_idx_temp].clone()
        num = int(num.item())
        if num <= 0:
            continue
        # 取出所有未标记结点，对cls_idx种类的预测概率
        # 上一轮的预测结果进行softmax，用作采样概率
        prev_out_local_prob = prev_out_local_prob.cpu()
        prob = prev_out_local_prob[:, cls_idx].squeeze()
        prob = get_similar(x_unlabeled, x_center, prob, a)
        # 按照预测概率进行采样
        # dst_idx_local = torch.multinomial(prob + 1e-12, num, replacement=False)
        value, dst_idx_local = torch.topk(prob, num)
        # 将采样的结点映射回全局索引
        dst_idx_local = dst_idx_local.to(unlabled_idx_temp.device)
        dst_idx = unlabled_idx_temp[dst_idx_local]
        # 增强训练集
        data_train_mask_temp[dst_idx] = True
        y_temp[dst_idx] = cls_idx
        # 将已经采样过的结点屏蔽，防止后续采样过程中继续采到，造成一个结点有不同的标签
        mask_prev = torch.ones(prev_out_local_prob.size(0), dtype=torch.bool)
        mask_prev[dst_idx_local] = False
        prev_out_local_prob = prev_out_local_prob[mask_prev]
        # 同步屏蔽unlabked_idx，用于结点正确映射回全局
        mask_unlabled_idx = torch.ones_like(unlabled_idx_temp, dtype=torch.bool)
        mask_unlabled_idx[dst_idx_local] = False
        unlabled_idx_temp = unlabled_idx_temp[mask_unlabled_idx]
        dst_idx_all.append(dst_idx)
        src_idx_all.append(cls_idx)

    dst_idx_all = [item for tensor in dst_idx_all for item in tensor.tolist()]

    return src_idx_all, dst_idx_all, data_train_mask_temp, y_temp
