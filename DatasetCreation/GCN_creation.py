import warnings
warnings.filterwarnings('ignore')
import torch
from torch_geometric.data import Data


if __name__=='__main__':
    """
    定义自己的数据集用于GCN
    """
    # 定义点的向量和标签
    x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.float)
    # 边的连接（无序）
    edge_index = torch.tensor([[0, 1, 2, 0, 3],
                               [1, 0, 1, 3, 2]], dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index)
    print(data)
