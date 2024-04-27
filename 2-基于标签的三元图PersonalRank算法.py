#撰写&创作：邓英帆(H-I-AM)
#诞生时间：2024/3/28 21:10

import time
import numpy as np
import pandas as pd
import math
import random
start_time = time.time()
#========================================



#引入delicious数据集，其中从左至右依次是user，item，label
# def genData():
#     data = pd.read_table('delicious_test/user_taggedbookmarks.dat', header=0, sep='\t')
#     data.drop(columns=['day','month','year','hour','minute','second'], inplace=True)
#     data.rename(columns={'userID': 'user', 'bookmarkID': 'item','tagID':'label'}, inplace=True)
#     data=data[:5000]
#     return data

def Predata():
    data = pd.read_table('delicious_test/user_taggedbookmarks.dat', header=0, sep='\t')
    data.drop(columns=['day', 'month', 'year', 'hour', 'minute', 'second'], inplace=True)
    data.rename(columns={'userID': 'user', 'bookmarkID': 'item', 'tagID': 'label'}, inplace=True)
    data = data[:10000]  # 取出前多少（用户，物品，标签），但都是序号表示
    # data2 = pd.read_table('delicious_test/bookmarkurl.dat', header=0, sep='\t')
    data1 = pd.read_excel('delicious_test/bookmarkurl.xlsx')
    data1.rename(columns={'id': 'item'}, inplace=True)
    data2 = pd.read_excel('delicious_test/tags.xlsx')

    # 将data中item全部由序号变为相应的值（也就是url）
    result = pd.DataFrame(np.full((data['item'].shape[-1], 1), ''))
    for i in range(len(data1.iloc[:, 0])):
        result.iloc[:, 0][data['item'] == data1.iloc[:, 0][i]] = data1.iloc[:, 1][i]
    data['item'] = result

    # 将data中的label全部由序号变为相应的值（就是value）
    result = pd.DataFrame(np.full((data['label'].shape[-1], 1), ''))
    for i in range(len(data2.iloc[:, 0])):
        result.iloc[:, 0][data['label'] == data2.iloc[:, 0][i]] = data2.iloc[:, 1][i]
    data['label'] = result

    data['user'] = [str(x) for x in data['user']]
    return data

#这两个函数构建了以（user，item）为键对，label为键值的字典，将对于相同user和item，不同的label全部收集起来形成一个list
def getuikey_label(data):
    UI_label=dict()
    for i in range(len(data)):
        lst=list(data.iloc[i])
        user=lst[0]
        item=lst[1]
        label=lst[2]
        addtodic(UI_label, (user, item), label)
    return UI_label
def addtodic(d, x, y):
    d.setdefault(x,[ ]).append(y)


#将输入的数据集划分训练集和测试集
def divata(Data, M, k, seed):
    # M:测试集占比；k:一个任意的数字，用来随机筛选测试集和训练集
    # train:训练集 test：测试集，都是字典，key是用户id,value是电影id集合
    data=Data.keys() ; test=[] ; train=[]
    #random.seed(seed)
    #注意这里我们先提取出来数据中的（user，item）
    # for user,item in data:
    #     if random.randint(0,M) < k :#or random.randint(0,M) == k-1:   #生成一个0-M的的随机数，看与k是否相等
    #         # 相等的概率是1/M，所以M即决定了测试集占比
    #         for label in Data[(user,item)]:
    #             test.append((user,item,label))
    #     else:
    #         for label in Data[(user, item)]:
    #             train.append((user,item,label))
    # return train,test
    for i, (user, item) in enumerate(data):
        if (i + 1) % 3 == 0:
            for label in Data[(user,item)]:  # 获取 (user, item) 对应的值作为 label
                test.append((user, item, label))
        else:
            for label in Data[(user,item)]:  # 获取 (user, item) 对应的值作为 label
                train.append((user, item, label))
    return train, test


#用于存储用户在测试集中的所有物品
def getTestuser_tags(user, test, N):
    tags=set()
    for user1,item,tag in test:
        if user1!=user:   #如果不是目标用户，就跳过
            continue
        if user1==user:   #如果是目标用户，就将该用户的item添加到items里去
            tags.add(tag)
    return list(tags)


#用于存储用户在测试集中的所有物品
def getTestuser_items(user, test, N):
    items=set()
    for user1,item,tag in test:
        if user1!=user:   #如果不是目标用户，就跳过
            continue
        if user1==user:   #如果是目标用户，就将该用户的item添加到items里去
            items.add(item)
    return list(items)


#获取给定用户在测试集中出现次数最多的前N个物品
def gettop_Testu(user, test, N):
    user_items=dict()
    for user1,item,tag in test:
        if user1!=user:   # user是我们的目标用户
            continue
        if user1==user:   #我们要选出来的就是目标用户的item
            if (user,item) not in user_items:
                user_items.setdefault((user,item),1)
            #如果是目标用户，但是用户没有用过item，就使用 setdefault() 方法来将该键添加到字典中，并将对应的值设置为1
            else:   #否则，直接给这样的数对＋1即可
                user_items[(user,item)]+=1
    testN=sorted(user_items.items(), key=lambda x: x[1], reverse=True)[0:N]   #对新形成的user_items中的值按照items出现的次数多少降序排列，同时取出前N个以及他们出现的次数
    items=[]
    for i in range(len(testN)):
        items.append(testN[i][0][1])
    #if len(items)==0:print "TU is None"
    return items   #testN是[((),_)]的形式，其中每个元组第一项是（user，item）对，第二项是该对出现的次数


#和之前基于随机游走的一模一样
def DoRecommendation(G, alpha, root, max_depth, N, user_items):
    rank = {x: 0 for x in G.keys()}
    rank[root] = 1
    for k in range(max_depth):
        tmp = {x: 0 for x in G.keys()}
        for i, ri in G.items():
            for j, wij in ri.items():
                tmp[j] += alpha * rank[i] * (wij / (1.0 * len(ri)))
        tmp[root] += (1 - alpha)
        rank = tmp
    lst=sorted(rank.items(),key=lambda x:x[1],reverse=True)
    # print(lst)
    items=[]
    for i in range(N):   #依旧是选取排名最高的N组拿出来放进item
        item=lst[i][0]
        item=str(item)
        if '/' in item and item not in user_items[root]:   #这里一定是推荐用户没有交互过的用品
            items.append(item)
    return items


def Recall(train,test,G,alpha,max_depth,N,user_items):
    hit=0   # 预测准确的数目
    total=0   # 所有行为总数
    for user,item,tag in train:
        tu=getTestuser_items(user, test, N)   #提取出该用户在测试集中交互过的物品
        rank=DoRecommendation(G, alpha, user, max_depth, N, user_items)
        # rank = [1,2,3,5,8,10]
        for item in rank:
            if item in tu:
                hit = hit+1
        total = total + len(tu)
    result = hit/(total*1.0)
    # result = ( random.random() * (max_depth/(max_depth+1)))/99
    # if hit > 0:
    print ("召回率存在，且为：",result)   #所有真正为正例的样本中，被分类器正确判定为正例的样本所占的比例。
    # else:
    #     print('no recommendation item appeared in test')
    return result


def Precision(train,test,G,alpha,max_depth,N,user_items):
    hit=0
    total=0
    for user, item, tag in train:
        tu = getTestuser_items(user, test, N)
        rank = DoRecommendation(G, alpha, user, max_depth, N, user_items)
        for item in rank:
            if item in tu:
                hit += 1
        total += N
    # result = ( random.random() * (max_depth/(max_depth+1)))/99
    # recall = recall / 45
    # print("精确度存在，且为：",result)
    print ("精确度存在，且为：",hit / (total * 1.0))   #分类器判定为正例的样本中，真正为正例的样本所占的比例
    return hit / (total * 1.0)


#覆盖率即为推荐过物品的数量除以总物品的数量
def Coverage(train,G,alpha,max_depth,N,user_items):
    recommend_items=set()
    all_items=set()
    for user, item, tag in train:
        all_items.add(item)   #注意前面用了集合，所以这里有重复的就会合并
        rank=DoRecommendation(G, alpha, user, max_depth, N, user_items)
        for item in rank:
            recommend_items.add(item)   #这里也是一样的道理，不会重复考虑
    print ("覆盖率存在，且为：",len(recommend_items)/(len(all_items)*1.0))
    return len(recommend_items)/(len(all_items)*1.0)


def Popularity(train,G,alpha,max_depth,N,user_items):
    item_popularity=dict()
    for user, item, tag in train:
        if item not in item_popularity:
            item_popularity[item]=0
        item_popularity[item] += 1
    ret=0   #用于存储最终的流行得分
    n=0   #用于记录参与流行度计算的物品的数量
    for user, item, tag in train:
        rank= DoRecommendation(G, alpha, user, max_depth, N, user_items)
        for item in rank:
            if item!=0 and item in item_popularity:
                ret+=math.log(1+item_popularity[item])
                n+=1
    if n==0:return 0.0
    ret/=n*1.0
    print ("流行度存在，且为：",ret)
    return ret


#基于余弦相似度计算物品相似度
def Cos(item_tags, item_i, item_j):
    ret = 0
    for b,wib in item_tags[item_i].items():
        if b in item_tags[item_j]:
            ret += wib*item_tags[item_j][b]
    ni=0
    nj=0
    for b,w in item_tags[item_i].items():
        ni+=w*w
    for b,w in item_tags[item_j].items():
        nj+=w*w
    if ret==0:
        return 0
    return ret/math.sqrt(ni*nj)


def Diversity(train,G,alpha,max_depth,N,user_items,item_tags):
    ret = 0.0
    n = 0
    for user, item, tag in train:
        rank = DoRecommendation(G, alpha, user, max_depth, N, user_items)
        for item1 in rank:
            for item2 in rank:
                if item1==item2:
                    continue
                else:
                    ret += Cos(item_tags, item1, item2)
                    n += 1
    print ("多样性存在，且为：",ret /(n*1.0))
    return ret /(n*1.0)


#构建用户-物品交互的图结构，并记录了用户对物品的操作情况，以及用户与标签、标签与物品之间的关联情况
def buildGraph(record):
    graph=dict() ; user_tags = dict() ; tag_items = dict() ; user_items = dict() ; item_tags = dict()
    #将 图，用户_标签,标签_商品，用户_商品，商品_标签 确定为字典结构
    for user, item, tag in record:
        if user not in graph:
            graph[user]=dict()
        if item not in graph[user]:
            graph[user][item]=1
        else:
            graph[user][item]+=1

        if item not in graph:
            graph[item]=dict()
        if user not in graph[item]:
            graph[item][user]=1
        else:
            graph[item][user]+=1

        if user not in user_items:
            user_items[user]=dict()
        if item not in user_items[user]:
            user_items[user][item]=1
        else:
            user_items[user][item]+=1

        if user not in user_tags:
            user_tags[user]=dict()
        if tag not in user_tags[user]:
            user_tags[user][tag]=1
        else:
            user_tags[user][tag]+=1

        if tag not in tag_items:
            tag_items[tag]=dict()
        if item not in tag_items[tag]:
            tag_items[tag][item]=1
        else:
            tag_items[tag][item]+=1

        if item not in item_tags:
            item_tags[item]=dict()
        if tag not in item_tags[item]:
            item_tags[item][tag]=1
        else:
            item_tags[item][tag]+=1

    return graph,user_items,user_tags,tag_items,item_tags

def Caculate(train, test, G, alpha, max_depth, N, user_items, item_tags):
    ##计算一系列评测标准

    recall=Recall(train,test,G,alpha,max_depth,N,user_items)
    precision=Precision(train,test,G,alpha,max_depth,N,user_items)
    coverage=Coverage(train,G,alpha,max_depth,N,user_items)
    popularity=Popularity(train,G,alpha,max_depth,N,user_items)
    diversity=Diversity(train,G,alpha,max_depth,N,user_items,item_tags)
    return recall,precision,coverage,popularity,diversity

if __name__=='__main__':
    new_data=Predata()
    UI_label = getuikey_label(new_data)
    # print(UI_label)
    (train, test) = divata(UI_label, 10, 3, 10)
    print(len(train));print(len(test))
    N=50;max_depth=30;alpha=0.8
    G, user_items, user_tags, tag_items, item_tags=buildGraph(train)
    recall, precision, coverage, popularity, diversity=Caculate(train, test, G, alpha, max_depth, N, user_items, item_tags)








#========================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序执行时间: {elapsed_time} 秒")
