import math
import numpy as np
import sys
import pprint
sys.setrecursionlimit(1000000)

#决策树类
class Decision_tree():
    def __init__(self,train_list,feature_dic,feature_distinct_value):
        self.train_list = train_list  #训练集列表数据
        self.train_size = len(train_list)
        self.feature_dic = feature_dic  #特征列表
        self.feature_size = len(feature_dic)
        train_set = list(range(0,self.train_size))  #训练集下标列表
        feature_set = list(range(0,self.feature_size))  #特征下标列表
        self.feature_distinct_value = feature_distinct_value  #特征不同取值列表
        self.threshold = 0  #阈值
        self.tree_root = self.build_ID3_tree(train_set,feature_set)  #ID3建树
        # self.tree_root = self.build_C45_tree(train_set,feature_set)  #C4.5建树
        # self.tree_root = self.build_CART_tree(train_set,feature_set)  #CART建树
        # pprint.pprint(self.tree_root)

    #划分数据集为小的数据集
    def split_dataset(self,train_set,feature,value):
        sub_data = []
        size = len(train_set)
        for i in range(size):
            if value == self.train_list[train_set[i]][feature]:
                #将每一个与value相同的值的下标加入到sub_data
                sub_data.append(train_set[i])
        return sub_data
    #计算熵
    def get_entropy(self,data_fre,size):
        entropy = 0.0
        for key,value in data_fre.items():  #key为取值名字，value为取值频数
            tmp = float(value) / size  #比例
            entropy += (-1) * tmp * math.log(tmp,2) #求熵公式
        return entropy
    #求信息增益
    def information_gain(self,train_set,feature):
        size = len(train_set)
        # 求出label中0和1的频数
        label_fre = self.get_label_fre(train_set)
        # 求经验熵
        empirical_entropy = self.get_entropy(label_fre,size)
        condition_entropy = 0.0
        sub_data_list = []
        data_size = len(self.feature_distinct_value[feature])
        #根据特征的不同取值，划分多个子数据集
        for i in range(data_size):
            #划分子数据集，是下标集合
            sub_data = self.split_dataset(train_set,feature,self.feature_distinct_value[feature][i])
            #如果子数据集存在，就加入到子数据集集合中
            if len(sub_data) > 0:
                sub_data_list.append(sub_data)

        size2 = len(sub_data_list)
        #对每个子数据集进行计算
        for i in range(size2):
            size3 = len(sub_data_list[i])
            #得出各种取值的频数
            local_fre = self.get_label_fre(sub_data_list[i])
            #求熵
            local_entropy = self.get_entropy(local_fre,size3)
            #加到条件熵中
            condition_entropy += (float(size3)/size) * local_entropy
            # print(condition_entropy)
        #得出信息增益
        information_gain = empirical_entropy - condition_entropy
        return information_gain
    #求出label中各种取值的频数
    def get_label_fre(self,train_set):
        label_fre = {}
        size = len(train_set)
        #遍历数据集
        for i in range(size):
            #如果key不存在，创建key，value加1
            if self.train_list[train_set[i]][6] not in label_fre.keys():
                label_fre[ self.train_list[train_set[i]][6] ] = 1
            else:
                label_fre[ self.train_list[train_set[i]][6] ] += 1
        return label_fre
    #求出特征中各种取值的频数
    def get_split_info_fre(self,train_set,feature):
        label_fre = {}
        size = len(train_set)
        #遍历数据集
        for i in range(size):
            #如果key不存在，创建key，value加1
            if self.train_list[train_set[i]][feature] not in label_fre.keys():
                label_fre[self.train_list[train_set[i]][feature]] = 1
            else:
                label_fre[self.train_list[train_set[i]][feature]] += 1
        return label_fre
    #创建ID3树
    def build_ID3_tree(self,train_set,feature_set):
        # 求出label的频数
        label_fre = self.get_label_fre(train_set)
        #如果特征已经选完，则返回label的众数为叶子结果
        if not feature_set:
            label_value = max(list(label_fre.items()) ,key = lambda x : x[1] )[0]
            return label_value
        #如果当前数据集中只有单一的label值，说明无须划分，返回结果
        if len(label_fre) == 1:
            label_value = list(label_fre.keys())
            return label_value[0]

        information_gain = {}
        feature_size = len(feature_set)
        train_size = len(train_set)
        #求各个特征的信息增益
        for i in range(feature_size):
            information_gain[feature_set[i]] = self.information_gain(train_set,feature_set[i])

        #求出信息增益最大的特征
        feature = max ( list(information_gain.items()) , key = lambda x : x[1] )[0]
        #如果信息增益小于0，说明划分没有意义，直接返回label众数结果
        if information_gain[feature] < self.threshold:
            label_value = max(list(label_fre.items()), key=lambda x: x[1])[0]
            return label_value

        #创建节点
        new_node = {}
        #存储特征名字
        new_node['feature'] = feature
        #存储数据集名字
        new_node['dataset'] = train_set
        sub_data_list = []
        sub_data_name = []
        data_size = len(self.feature_distinct_value[feature])
        #依次根据feature取值划分
        for i in range(data_size):
            value = self.feature_distinct_value[feature][i]
            # 存储子数据集名字
            sub_data_name.append(value)
            #划分子数据集
            sub_data = self.split_dataset(train_set,feature,value)
            if len(sub_data) > 0 :
                sub_data_list.append(sub_data)

        sub_data_size = len(sub_data_list)
        #子特征集下标列表
        sub_feature_set = feature_set[:]
        sub_feature_set.remove(feature)

        #对每个子数据集，递归调用建树函数，接受结果作为子节点
        for k in range(sub_data_size):
            new_node[sub_data_name[k]] = self.build_ID3_tree(sub_data_list[k],sub_feature_set)
        #返回根节点
        return new_node

    def build_C45_tree(self,train_set,feature_set):
        # 求出label的频数
        label_fre = self.get_label_fre(train_set)
        # 如果特征已经选完，则返回label的众数为叶子结果
        if not feature_set:
            label_value = max(list(label_fre.items()) ,key = lambda x : x[1] )[0]
            return label_value
        # 如果当前数据集中只有单一的label值，说明无须划分，返回结果
        if len(label_fre) == 1:
            label_value = list(label_fre.keys())
            return label_value[0]

        information_gain = {}
        feature_size = len(feature_set)
        train_size = len(train_set)
        # 求各个特征的信息增益率
        for i in range(feature_size):
            information_gain[feature_set[i]] = self.information_gain(train_set,feature_set[i])
            feature_fre = self.get_split_info_fre(train_set,feature_set[i])
            split_encropy = self.get_entropy(feature_fre,train_size)
            if split_encropy == 0:
                split_encropy += 0.5
            information_gain[feature_set[i]] = information_gain[feature_set[i]] / split_encropy

        # 求出信息增益率最大的特征
        feature = max ( list(information_gain.items()) , key = lambda x : x[1] )[0]

        if information_gain[feature] < self.threshold:
            label_value = max(list(label_fre.items()), key=lambda x: x[1])[0]
            return label_value

        # 创建节点
        new_node = {}
        #存储特征名字
        new_node['feature'] = feature
        #存储数据集
        new_node['dataset'] = train_set
        sub_data_list = []
        sub_data_name = []
        data_size = len(self.feature_distinct_value[feature])
        # 依次根据feature取值划分
        for i in range(data_size):
            value = self.feature_distinct_value[feature][i]
            #存储子数据集名字
            sub_data_name.append(value)
            #划分子数据集
            sub_data = self.split_dataset(train_set,feature,value)
            if len(sub_data) > 0 :
                sub_data_list.append(sub_data)

        sub_data_size = len(sub_data_list)
        #子特征集下标列表
        sub_feature_set = feature_set[:]
        sub_feature_set.remove(feature)
        # 对每个子数据集，递归调用建树函数，接受结果作为子节点
        for k in range(sub_data_size):
            new_node[sub_data_name[k]] = self.build_C45_tree(sub_data_list[k],sub_feature_set)
        #返回根节点
        return new_node
    #求条件gini指数
    def get_condition_gini(self,label_fre,size):
        condition_gini = 1.0
        for key,value in label_fre.items():
            #求gini公式
            condition_gini -= (float(value) / size)*(float(value) / size)
        return condition_gini
    #求gini指数
    def get_gini(self,train_set,feature):
        train_size = len(train_set)
        data_size = len(self.feature_distinct_value[feature])
        gini = 0.0
        #根据feature的取值划分子数据集，对每个子数据集求gini
        for i in range(data_size):
            #划分子数据集
            sub_data = self.split_dataset(train_set,feature,self.feature_distinct_value[feature][i])
            if len(sub_data) > 0:
                size = len(sub_data)
                #求各个取值的频数
                label_fre = self.get_label_fre(sub_data)
                #求该条件下的gini指数
                condition_gini = self.get_condition_gini(label_fre,size)
                #求gini指数公式
                gini += (float(size)/train_size) * condition_gini

        return gini

    def build_CART_tree(self,train_set,feature_set):
        # 求出label的频数
        label_fre = self.get_label_fre(train_set)
        # 如果特征已经选完，则返回label的众数为叶子结果
        if not feature_set:
            label_value = max(list(label_fre.items()) ,key = lambda x : x[1] )[0]
            return label_value
        # 如果当前数据集中只有单一的label值，说明无须划分，返回结果
        if len(label_fre) == 1:
            label_value = list(label_fre.keys())
            return label_value[0]

        gini = {}
        feature_size = len(feature_set)
        train_size = len(train_set)
        # 求各个特征的gini指数
        for i in range(feature_size):
            gini[feature_set[i]] = self.get_gini(train_set,feature_set[i])

        # 求出gini指数最小的特征
        feature = min ( list(gini.items()) , key = lambda x : x[1] )[0]

        # 创建节点
        new_node = {}
        #存储特征名字
        new_node['feature'] = feature
        #存储数据集
        new_node['dataset'] = train_set
        sub_data_list = []
        sub_data_name = []
        data_size = len(self.feature_distinct_value[feature])
        # 依次根据feature取值划分
        for i in range(data_size):
            value = self.feature_distinct_value[feature][i]
            #存储子数据集名字
            sub_data_name.append(value)
            #划分子数据集
            sub_data = self.split_dataset(train_set,feature,value)
            if len(sub_data) > 0 :
                sub_data_list.append(sub_data)

        sub_data_size = len(sub_data_list)
        #子特征集下标列表
        sub_feature_set = feature_set[:]
        sub_feature_set.remove(feature)
        # 对每个子数据集，递归调用建树函数，接受结果作为子节点
        for k in range(sub_data_size):
            new_node[sub_data_name[k]] = self.build_CART_tree(sub_data_list[k],sub_feature_set)
        #返回根节点
        return new_node
    #预测1
    def classify(self,test_list):
        node = self.tree_root #根节点
        #调用递归函数查找预测
        return self.node_classify(node,test_list)
    #递归预测分类
    def node_classify(self,node,test_list):
        #遇到子节点，返回结果
        if type(node) != dict:
            return node
        #求当前节点的特征名字
        feature_value = test_list[node['feature']]
        #如果特征值不存在，直接返回label众数
        if feature_value not in node.keys():
            label_fre = self.get_label_fre(node['dataset'])
            return max(list(label_fre.items()) ,key = lambda x : x[1] )[0]
        #递归调用进行下一步的预测分类
        return self.node_classify(node[feature_value],test_list)




