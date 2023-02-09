from CVSOperator import CSVOperator
import os
import random as rd
data_file = "dataset\\diabetes\\pre_operate\\step1\\step1.csv"
out_data_document = "dataset\\diabetes\\pre_operate\\step2\\"
label = [0, 1]  # 0
max_HighBP = 1  # 1
max_HighChol = 1  # 2
max_CholCheck = 1  # 3
max_BIM = 98  # 4
max_Smoker = 1  # 5
max_Stroke = 1  # 6
max_HeartDiseaseorAttack = 1  # 7
max_PhysActivity = 1  # 8
max_Fruits = 1  # 9
max_Veggies = 1  # 10
max_HvyAlcoholConsump = 1  # 11
max_AnyHealthcare = 1  # 12
max_NoDocbcCost = 1  # 13
max_GenHlth = 5  # 14
max_MentHlth = 30  # 15
max_PhysHlth = 30  # 16
max_DiffWalk = 1  # 17
max_Sex = 1  # 18
max_Age = 13  # 19
max_Education = 6  # 20
max_Income = 8  # 21
norm_list = [max_HighBP, max_HighChol, max_CholCheck, max_BIM, max_Smoker, max_HeartDiseaseorAttack,
             max_PhysActivity, max_Fruits, max_Veggies, max_HvyAlcoholConsump, max_AnyHealthcare,
             max_NoDocbcCost, max_GenHlth, max_MentHlth, max_PhysHlth, max_DiffWalk, max_Sex, max_Age,
             max_Education, max_Income]


def step1():
    # 归一化
    csv_reader = CSVOperator(data_file, 'r')
    new_row = []
    c = 0
    for row in csv_reader.reader:
        if c == 0:
            c = c + 1
            continue
        cut = 1
        for i in range(1, len(row)):
            if i == 19:
                cut = 2
                continue
            row[i] = float(float(row[i])/norm_list[i-cut])
        new_row.append(row)
    csv_reader.end()
    csv_writer = CSVOperator(out_data_document + "/step1/step1.csv", 'w')
    csv_writer.write_row(new_row)


def step2():
    # 分发数据模拟联邦学习,并进行随机的特征分配
    # 选择6个全局共有特征，剩余特征进行随机删除
    # 0 是标签
    global_feature = [0, 1, 14, 6, 2, 12, 18]
    csv_reader = CSVOperator(data_file, 'r')
    local_feature_list = []
    new_row_4_write = []
    length = 22
    csv_writers = []
    for i in range(max_Age):
        csv_writers.append(CSVOperator(out_data_document + str(i) + '.csv', 'w'))
        local_feature_num = rd.randint(len(global_feature)-1, length-2)
        local_feature_temp = []
        for j in range(local_feature_num):
            d = 0
            while d in global_feature:
                d = rd.randint(0, 22)
            local_feature_temp.append(d)
        local_feature_list.append(local_feature_temp)
        temp = []
        new_row_4_write.append(temp)
    for row in csv_reader.reader:
        # label = int(row[-1])
        new_row_temp = []
        for i in range(len(row)):
            if i in local_feature_list[int(float(row[19]))-1] or i in global_feature:
                new_row_temp.append(str(row[i]))
            else:
                new_row_temp.append('-')
        new_row_4_write[int(float(row[19]))-1].append(new_row_temp)
    for i in range(len(new_row_4_write)):
        csv_writers[i].write_row(new_row_4_write[i])

def run():
    # step1()
    step2()





