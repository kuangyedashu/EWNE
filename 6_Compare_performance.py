# import networkx as nx
# import numpy as np
# import matplotlib.pyplot as plt
#calculate the overlap number of the complex i get with the known conplex
#file=open("dip_complex_degree.txt")
#file存放的计算得到的结果，file1存放的是标准功能模块
str_result="策略验证实验/GAT权重置零final_krogan_output.txt"
file=open(str_result)
file1=open("dataset/Form_CYC20083.txt")
# file1=open("dataset/golden_standard.txt")

# g=nx.Graph() 
#下面两个变量分别存储（1）预测的模块的数量（2）标准的模块的数量
predicted_num=len(file.readlines())
reference_num=len(file1.readlines())
file.close()
file1.close()

file=open(str_result)
file1=open("dataset/Form_CYC20083.txt")
# file1=open("dataset/golden_standard.txt")
#下面这两个变量分别存储（1）标准模块的值（2）预测模块的值
reference_complex=[]
for j in file1:
    j = j.rstrip()
    j = j.rstrip('\n')
    complex_list = j.split('	')
    reference_complex.append(complex_list)

predicted_complex=[]
for i in file:
    i = i.rstrip()
    i = i.rstrip('\n')
    node_list = i.split(' ')
    predicted_complex.append(node_list)
#下面的是各个评测指标。
#precision
number=0
c_number=0
row=1
for i in predicted_complex:
    overlapscore=0.0
    for j in reference_complex:
        set1 = set(i)
        set2 = set(j)
        overlap = set1 & set2
        score = float((pow(len(overlap), 2))) / float((len(set1) * len(set2)))
        if (score > overlapscore):
            overlapscore = score
    if (overlapscore > 0.2):
            number = number + 1
            print (row)
            print (" ")
    row=row+1
#recall
for i in reference_complex:
    overlapscore=0.0
    for j in predicted_complex:
        set1 = set(i)
        set2 = set(j)
        overlap = set1 & set2
        score = float((pow(len(overlap), 2))) / float((len(set1) * len(set2)))
        if(score > overlapscore):
            overlapscore = score
    if(overlapscore > 0.25):
        c_number=c_number+1
#sn
T_sum1=0.0
N_sum=0.0
for i in reference_complex:
    max=0.0
    for j in predicted_complex:
        set1=set(i)
        set2=set(j)
        overlap=set1&set2
        if len(overlap)>max:
            max=len(overlap)
    T_sum1=T_sum1+max
    N_sum=N_sum+len(set1)
#ppv
T_sum2=0.0
T_sum=0.0
for i in predicted_complex:
    max=0.0
    for j in reference_complex:
        set1=set(i)
        set2=set(j)
        overlap=set1&set2
        T_sum=T_sum+len(overlap)
        if len(overlap)>max:
            max=len(overlap)
    T_sum2=T_sum2+max
#cv
cv_son,cv_mother = 0, 0
proteins_son = []
for complex in predicted_complex:
    for protein in complex:
        proteins_son.append(protein)
proteins_son = list(set(proteins_son))
cv_son = len(proteins_son)


print ("\n")
print (number,predicted_num)# matched predicted complex number
#print c_number,reference_num# matched reference complex number
precision=float(number/float(predicted_num))
recall=float(c_number/float(reference_num))
F1=float((2*precision*recall)/(precision+recall))
Sn=float(T_sum1)/float(N_sum)
PPV=float(T_sum2)/float(T_sum)
Acc=pow(float(Sn*PPV),0.5)

precision, recall, F1, Sn, PPV, Acc = round(precision, 3), round(recall, 3), round(F1, 3), round(Sn, 3), round(PPV, 3), round(Acc, 3)
print ("precision+F1--ACC+F1--precision recall F1 Sn PPV Acc cv_son ")
print (precision+F1, Acc+F1 ,precision, recall, F1, Sn, PPV, Acc, cv_son)

file.close()
file1.close()