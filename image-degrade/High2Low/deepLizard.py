
#callable neural network
import torch

in_features = torch.tensor([1,2,3], dtype=torch.float32)
weight_matrix = torch.tensor([
    [1,2,3,4],
    [4,18,32,90],
    [5,20,77,4],
    [6,6,3,48],
], dtype=torch.float32)

weight_matric.matmul(in_fetures)


## is equivalent to

fc = nn.Linear(in_features=4, out_features=3, bias=False)
print(fc(in_features))

##but weights are changed

##explicitly setting the weights
fc.weight = nn.Parameter(weight_matrix)
fc(in_features)


# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self