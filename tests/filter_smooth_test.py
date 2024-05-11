import os
import sys

import numpy as np
currentdir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(currentdir)
from algorithms.machine_learning.feature_extraction.filter_smooth import *
if __name__ == '__main__':
    data = np.random.random(100)
    # mode = 'mv'
    mode = 'extremum'
    filter_data_wave = FilterSmooth(mode, **{'segment_num':20}).fit(data)[0:len(data)]
    print(f"data{data}")
    print(f"filter_data_wave{filter_data_wave}")
