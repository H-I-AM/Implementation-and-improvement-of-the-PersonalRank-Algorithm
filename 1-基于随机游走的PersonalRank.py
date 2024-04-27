#撰写&创作：邓英帆(H-I-AM)
#诞生时间：2024/3/28 21:01

import time
start_time = time.time()
#========================================

def PersonalRank(G, alpha, root, max_depth):
    rank = {x: 0 for x in G.keys()} ; rank[root] = 1
    # 随机游走计算最终概率
    for k in range(max_depth):
        tmp = {x: 0 for x in G.keys()}   #暂时将推荐的物品存在这个里面，作为迭代过程和rank之间的中转站
        for i, ri in G.items():   #取出节点和他所有的出边指向的节点ri，开始游走
            for j, wij in ri.items():
                tmp[j] += alpha * rank[i] / (1.0 * len(ri))   #边的权重都为1，利用出边的数量归一化得到1/len(ri)
        tmp[root] += (1 - alpha)   #也有一部分概率不向下游走
        rank = tmp
    lst = sorted(rank.items(), key=lambda x: x[1], reverse=True)   #从高到低对推荐物品进行排序
    for ele in lst:
        print("%s:%.3f, \t" % (ele[0], ele[1]))   #输出推荐物品以及相应的概率值（推荐值）
    return rank


if __name__ == '__main__':
    alpha = 0.8
    G = {'A': {'a': 1, 'c': 1},
         'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
         'C': {'c': 1, 'd': 1},
         'a': {'A': 1, 'B': 1},
         'b': {'B': 1},
         'c': {'A': 1, 'B': 1, 'C': 1},
         'd': {'B': 1, 'C': 1}}
    PersonalRank(G, alpha, 'b', 50)


# if __name__=='__main__':
#     alpha=0.75
    # G={'8': {'a': 1, 'b': 3,},
    #  'a': {'8': 1},
    #  'b': {'8': 3},
    #  'c': {'8': 3},
    #  'd': {'8': 2},
    #  'e': {'8': 2},
    #  'f': {'8': 3}}

    # G = {'A': {'a': 1, 'c': 1},
    #      'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
    #      'C': {'c': 1, 'd': 1},
    #      'a': {'A': 1, 'B': 1},
    #      'b': {'B': 1},
    #      'c': {'A': 1, 'B': 1, 'C': 1},
    #      'd': {'B': 1, 'C': 1}}
    # PersonalRank(G, alpha, 'B', 20)

#========================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序执行时间: {elapsed_time} 秒")
