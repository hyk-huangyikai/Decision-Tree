import math
import numpy as np
import sys
import csv
from decision_tree import Decision_tree

# sys.setdefaultencoding('utf-8')

#处理数据
def manage_data(train_list):
    feature_distinct_value = {}  #存储各个特征的所有不同的取值
    feature_dic = {}  #存储各个特征的所有数据（包括重复的）
    for j in range(6): #不重复
        feature_dic[j] = []
        feature_distinct_value[j] = []

    train_size = len(train_list)
    # 遍历每一行输入的数据
    for i in range(train_size):
        for j in range(6):
            # 加入数据到对应的特征
            feature_dic[j].append(train_list[i][j])
            if train_list[i][j] not in feature_distinct_value[j]:
                # 加入不同的数据到对应的特征
                feature_distinct_value[j].append(train_list[i][j])

    return feature_dic,feature_distinct_value


def main():
    file_name1 = 'lab2_data/Car_train.csv'
    # file_name1 = 'test1.csv'
    with open (file_name1,'r',encoding='utf-8') as csv_file1:   #读取训练文件
        csv_reader1 = csv.reader(csv_file1)
        train_list1 = list(csv_reader1)

    file_name2 = 'lab2_data/Car_test.csv'
    # file_name2 = 'lab2_data/Car_verification.csv'
    # file_name2 = 'lab2_data/Car_train.csv'
    # file_name2 = 'test2.csv'
    with open (file_name2,'r',encoding='utf-8') as csv_file2:
        csv_reader2 = csv.reader(csv_file2)
        test_list = list(csv_reader2)

    feature_dic, feature_distinct_value = manage_data(train_list1)  # 将数据进行处理
    decision_tree = Decision_tree(train_list1,feature_dic,feature_distinct_value)  #进行决策树建树

    correct = 0
    test_size = len(test_list)
    csv_file3 = open('lab2_data/16337098_huangyikai.csv','w',newline='')
    csv_write = csv.writer(csv_file3)
    for i in range(test_size):  #测试每个样例
        test_value = decision_tree.classify(test_list[i])  #调用分类预测函数，得出预测结果
        # if test_value == test_list[i][6]:  #如果预测结果与样本结果相同，则准确数目加1
        #     correct += 1
        csv_write.writerow([test_value])
        # print(test_value)

    # print("Accuracy: ",float(correct) / test_size)  #输出准确率


    # rate_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    # k_value = 10
    # total_count = len(train_list1)
    # local_count = int(total_count / k_value)
    # wirte_file = open ('Accuracy.txt','w',encoding='utf-8')
    # for k in range(k_value):
    #     index1 = local_count * k
    #     index2 = local_count * (k+1)
    #     test_list = train_list1[index1:index2]
    #     if index1 != 0:
    #         train_list = train_list1[0:index1]  #训练集列表
    #         train_list2 = train_list1[index2:]  #验证集列表
    #         train_list.extend(train_list2)
    #     else:
    #         train_list = train_list1[index2:]
    #     train_num  = len(train_list)
    #     verification_num = len(test_list)
    #     print("total_count: ", total_count, "train_num: ", train_num, "verification_num: ", verification_num)
    #     feature_dic,feature_distinct_value = manage_data(train_list)  #将数据进行处理
    #     decision_tree = Decision_tree(train_list,feature_dic,feature_distinct_value)  #进行决策树建树
    #
    #     correct = 0
    #     test_size = len(test_list)
    #     for i in range(test_size):  #测试每个样例
    #         test_value = decision_tree.classify(test_list[i])  #调用分类预测函数，得出预测结果
    #         if test_value == test_list[i][6]:  #如果预测结果与样本结果相同，则准确数目加1
    #             correct += 1
    #
    #     print("k: ",str(k),"  Accuracy: ",float(correct) / test_size)  #输出准确率
    #     str_name =  str(float(correct) / test_size) + ","
    #     wirte_file.write(str_name)

if __name__ == "__main__":
    main()