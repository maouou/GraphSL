import torch
from scipy.spatial.distance import cosine


tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([2.0, 3.0, 4.0])
tensor3 = torch.tensor([3.0, 4.0, 5.0])
tensor4 = torch.tensor([4.0, 5.0, 6.0])
tensor5 = torch.tensor([5.0, 6.0, 7.0])
tensor6 = torch.tensor([1.0, 2.0, 3.0])
tensor7 = torch.tensor([2.0, 3.0, 4.0])
tensor8 = torch.tensor([3.0, 4.0, 5.0])
tensor9 = torch.tensor([4.0, 5.0, 6.0])
tensor10 = torch.tensor([5.0, 6.0, 7.0])

x = torch.stack(tensors=[tensor1, tensor2, tensor3, tensor4, tensor5, tensor6,tensor7, tensor8,tensor9, tensor10])
train_mask = torch.tensor([True, True, True, True, False, True, True, True, True, True])
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

x_temp = x[train_mask]
y_temp = y[train_mask]
value1 = 0
value2 = 1

cls_indices = torch.nonzero(y_temp == value2, as_tuple=True)
cls_indices = cls_indices[0].tolist()
x_all = x_temp[cls_indices]
x_mean = x_all.mean(dim=0)
print(x_mean)


cosine_similarity_all = []
# 计算点积
for tensor_temp in x_all:
    similarity = 1 - cosine(tensor_temp, x_mean)
    cosine_similarity_all.append(similarity)

print(cosine_similarity_all)  # 将张量转为 Python 标量