import os
import sys
current_dir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(current_dir)
from algorithms.machine_learning.hmm.hmm_model import *


if __name__ == '__main__':
    # 假设你的数据集是X，它是一个二维数组，每一行是一个观察
    np.random.seed(42)
    data = np.random.randn(100, 5)  # 100个时间步，每个时间步有5个特征
    hidden_states = initialize_hmm_with_gmm(data)
    logprob, hidden_states1 = gmm_hmm(data)
    print(hidden_states)
    print(logprob, hidden_states1)
