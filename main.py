import os.path as osp
import tqdm
import random
import numpy as np
import torch
import torch.nn.functional as F

from args import parse_args
from data_utils import get_dataset, get_idx_info, make_longtailed_data_remove, get_step_split
from gens import graph_augment, neighbor_sampling, duplicate_neighbor, saliency_mixup, sampling_idx_individual_dst, cout_KDE
from nets import create_gcn, create_gat, create_sage
from utils import CrossEntropy
from sklearn.metrics import balanced_accuracy_score, f1_score
from neighbor_dist import get_PPR_adj
from pretrain import pre_train

import warnings

warnings.filterwarnings("ignore")


def train():
    global class_num_list, idx_info, prev_out
    global data_train_mask, data_val_mask, data_test_mask
    model.train()
    optimizer.zero_grad()
    prev_out_local = prev_out.clone()[unlabeled_idx]
    prev_last = prev_out.clone()
    sampling_src_idx, sampling_dst_idx, new_data_train_mask, new_y = graph_augment(data.x, class_num_list, prev_out_local, prev_last,
                                                                                       unlabeled_idx, data_train_mask,
                                                                                       data.y, args.a, args.tau, args.max)
    # 将新采样获得的结点对应边加入其中
    new_edge_index = neighbor_sampling(data.edge_index, train_edge_mask, sampling_dst_idx)
    # 未生成新节点，不需要对数据集进行扩充
    new_x = data.x
    output = model(new_x, new_edge_index)
    prev_out = output.clone()
    criterion(output[new_data_train_mask], new_y[new_data_train_mask]).backward()

    with torch.no_grad():
        model.eval()
        output = model(data.x, data.edge_index[:, train_edge_mask])
        val_loss = F.cross_entropy(output[data_val_mask], data.y[data_val_mask])
    optimizer.step()
    scheduler.step(val_loss)
    return


@torch.no_grad()
def test():
    model.eval()
    logits = model(data.x, data.edge_index[:, train_edge_mask])
    accs, baccs, f1s = [], [], []
    y_count = []
    y_correct = []
    for mask in [data_train_mask, data_val_mask, data_test_mask]:
        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
        y_count_temp = np.zeros(7)
        y_correct_temp = np.zeros(7)
        for idx, y_temp in enumerate(y_true):
            y_count_temp[y_temp] = y_count_temp[y_temp] + 1
            if pred[idx] == y_temp:
                y_correct_temp[y_temp] = y_correct_temp[y_temp] + 1
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)
        y_count.append(y_count_temp)
        y_correct.append(y_correct_temp)
    return accs, baccs, f1s, y_count, y_correct

args = parse_args()
seed = args.seed
device = torch.device(args.device)

torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)
print(args.a)

path = args.data_path
path = osp.join(path, args.dataset)
dataset = get_dataset(args.dataset, path, split_type='full')
data = dataset[0]
n_cls = data.y.max().item() + 1
data = data.to(device)

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()
    # stats: 训练集标签
    stats = data.y[data_train_mask]
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    # idx_info: 训练集每个类的index分布
    idx_info = get_idx_info(data.y, n_cls, data_train_mask)
    class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
        make_longtailed_data_remove(data.edge_index, data.y, n_data, n_cls, args.imb_ratio, data_train_mask.clone())
    train_idx = data_train_mask.nonzero().squeeze()
    all_indices = list(range(len(data.y)))
    # 未标记节点的索引
    unlabeled_idx = [idx for idx in all_indices if idx not in train_idx]

    labels_local = data.y.view([-1])[train_idx]
    train_idx_list = train_idx.cpu().tolist()
    local2global = {i: train_idx_list[i] for i in range(len(train_idx_list))}
    global2local = dict([val, key] for key, val in local2global.items())
    idx_info_list = [item.cpu().tolist() for item in idx_info]
    idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in idx_info_list]

else:
    raise NotImplementedError

labels_local = data.y.view([-1])[train_idx]
train_idx_list = train_idx.cpu().tolist()
local2global = {i: train_idx_list[i] for i in range(len(train_idx_list))}
global2local = dict([val, key] for key, val in local2global.items())
idx_info_list = [item.cpu().tolist() for item in idx_info]
idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in idx_info_list]
neighbor_dist_list = get_PPR_adj(data.x, data.edge_index[:, train_edge_mask], alpha=0.05, k=128, eps=None)

if args.net == 'GCN':
    model = create_gcn(nfeat=dataset.num_features, nhid=args.feat_dim, nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
elif args.net == 'GAT':
    model = create_gat(nfeat=dataset.num_features, nhid=args.feat_dim, nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
elif args.net == "SAGE":
    model = create_sage(nfeat=dataset.num_features, nhid=args.feat_dim, nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
else:
    raise NotImplementedError("Not Implemented Architecture!")

model = model.to(device)
criterion = CrossEntropy().to(device)

optimizer = torch.optim.Adam(
    [dict(params=model.reg_params, weight_decay=5e-4), dict(params=model.non_reg_params, weight_decay=0), ], lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=False)

best_val_acc_f1 = 0
saliency = None

prev_out = pre_train()

for epoch in tqdm.tqdm(range(args.epoch)):
    train()
    accs, baccs, f1s, y_count, y_correct = test()
    train_acc, val_acc, tmp_test_acc = accs
    train_f1, val_f1, tmp_test_f1 = f1s
    val_acc_f1 = (val_acc + val_f1) / 2.
    if val_acc_f1 > best_val_acc_f1:
        best_val_acc_f1 = val_acc_f1
        test_acc = accs[2]
        test_bacc = baccs[2]
        test_f1 = f1s[2]
        test_y_count = y_count[2]
        test_y_correct = y_correct[2]
        train_y_count = y_count[0]
        train_y_correct = y_correct[0]

print('acc: {:.2f}, bacc: {:.2f}, f1: {:.2f}'.format(test_acc * 100, test_bacc * 100, test_f1 * 100))
print("test_y_count:", test_y_count)
print("test_y_correct:", test_y_correct)
print("train_y_count:", train_y_count)
print("train_y_correct:", train_y_correct)
