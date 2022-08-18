import numpy as np

str0="策略验证实验/GAT权重置零krogan_GAT_W0_rewrite.txt"#网络嵌入向量
str1="dataset/krogan2006core.txt"#相互作用数据
str2="策略验证实验/GAT权重置零/krogan_GAT_W0_nodevector.txt"#file2中，存放的格式：“nodei vectori”
str3="策略验证实验/GAT权重置零/krogan_GAT_W0_sim.txt"

file0=open(str0)
file1=open(str1)
file2=open(str2,'w')
print ("get the vector representation: ")
#node存放相互作用数据中的所有节点
node=[]
for j in file1:
    temp1=j.split('	')[0]
    temp2=j.split('	')[1].rstrip('\n')
    if temp1 not in node:
        node.append(temp1)
    if temp2 not in node:
        node.append(temp2)
file1.close()

d=[]
for i in file0:
    d.append(i)
file0.close()
for i in range(len(node)):
    file2.write(node[i])
    file2.write(' ')
    file2.write(d[i])
    file2.write('\n')
file2.close()
print ("calculate the similarity between two nodes:")

file1=open(str1)#蛋白质相互作用
file2=open(str2)#224维度的特征，格式为：nodei vector1
file3=open(str3,'w')#余弦相似度

def cos_sim(vector1,vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    vector1 = vector1[0].split()
    xx = []
    for i in vector1:
        tmp = i[0:-1]
        xx.append(tmp)
    vector11 = []
    for n in xx:
        vector11.append(float(n))
    vector2 = vector2[0].split()
    xx = []
    for i in vector2:
        tmp = i[0:-1]
        xx.append(tmp)
    vector22 = []
    for n in xx:
        vector22.append(float(n))
    # vector1 = vector1[0].split()
    # vector1=list(map(int,vector1))
    # vector2 = vector2[0].split()
    # vector2=list(map(int,vector2))
    for a, b in zip(vector11, vector22):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    result=dot_product / ((normA * normB) ** 0.5)
    return result
#edge_name_name 和 vectorh这两个变量，应该是用来存放格式化后的数据的。（因为存放在文件中的数据，格式不太对）
edge_name_name=[]
for i in file1:
    node_name1=i.split('	')[0]
    node_name2=i.split('	')[1]
    node_name2 = node_name2.split('\n')[0]
    d={}
    d['node_name1']=node_name1
    d['node_name2']=node_name2
    edge_name_name.append(d)

vector=[]
for i in file2:
    if not i.strip(): continue
    node_name=i.split(' ',1)[0]
    node_vector=i.split(' ',1)[1].rstrip('\n')
    node_vector = node_vector.split('	')
    # node_vector = map(float, node_vector)
    d = {}
    d['node_name'] = node_name
    d['node_vector'] = node_vector
    vector.append(d)
#下面就是写入文件中，格式为：“nodei nodej float”
v1=[]
v2=[]
for i in edge_name_name:
    temp1=0
    temp2=0
    for j in vector:#vector中的格式为{[key,value],[key,value],[key,value]}
        if(i['node_name1']==j['node_name']):
            v1 = j['node_vector']
            # v1=np.array(j['node_vector'])
            temp1=1
    for z in vector:
        if(i['node_name2']==z['node_name']):
            v2 = z['node_vector']
            # v2=np.array(z['node_vector'])
            temp2=1
    if(temp1==1)and(temp2==1):
        result=cos_sim(v1,v2)
        file3.write(i['node_name1'])
        file3.write(' ')
        file3.write(i['node_name2'])
        file3.write(' ')
        file3.write(str(result))
        file3.write('\n')


file1.close()
file2.close()
file3.close()