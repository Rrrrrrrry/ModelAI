from collections import Counter
from itertools import chain


def compute_similarity(list1, list2, threshold=0.7):
    """
    计算两段数据的相似度
    :param list1:
    :param list2:
    :param threshold:
    :return:
    """
    if len(list1) == 0 or len(list2) == 0:
        return False
    counter1 = Counter(list1)
    counter2 = Counter(list2)
    common_elements = counter1 & counter2
    # return (sum(common_elements.values()) / len(list1)) > threshold
    return (sum(common_elements.values()) / min(len(list1), len(list2))) > threshold


def merge_classifications(data, windows=24, sim_threshold=0.7):
    """
    将分类结果进行合并
    :param data:
    :param windows:
    :param sim_threshold:
    :return:
    """
    """
    将相似的两段数据合并
    """
    data = list(data)
    maps = []
    start = 0
    end = start + windows
    que = []
    while start < len(data):
        first = data[start:end]
        last = data[end:end + windows]
        que = que + first
        if not compute_similarity(first, last, sim_threshold):
            # 将一段数据修改为相同的名字
            que = [max(que, key=Counter(que).get)] * len(que)
            maps.append(que)
            que = []
        start = end
        end = start + windows
    print(f"小窗口合并")
    print(f"maps{maps}")
    """
    小窗口合并
    """
    new_maps = []
    que = []
    for i, item in enumerate(maps):
        if len(item) != windows:
            if que != []:
                que = que + item
                new_maps.append(que)
                que = []
            else:
                new_maps.append(item)
        else:
            que = que + item
            if i == len(maps) - 1:
                new_maps.append(que)
    """
    将数据展开
    """
    new_maps = list(chain.from_iterable(new_maps))
    return new_maps

# if __name__ == '__main__':
#     data = [1, 2, 3, 1, 2, 2]
#     data_merge = merge_classifications(data, windows=3, sim_threshold=0.7)
#     print(f"data_merge{data_merge}")