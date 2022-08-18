#根据原始数据，生成对应的0、1形式的矩阵（邻接矩阵和属性矩阵）
import numpy as np
import re
str1="dataset/krogan2006core_noise_7000.txt"#dataset/krogan2006extended.txt
str2="matrix/Network_krogan2006core_noise_7000.txt"#matrix/Network_krogan2006extended.txt
str3="matrix/Attribute_krogan2006core_noise_7000.txt"#matrix/Attribute_krogan2006extended.txt

#file1是相互作用数据 file2是要生成的0、1相互作用数据 filetemp存放的是file1所有的节点
file1=open(str1)
file2=open(str2,'w')
filetemp=open("dataset/krogan2006core_node.txt",'w')#"krogan2006extended_node.txt",'w'
print ("create topological network!")
#node存放file1中的所有节点 filetemp也是
node=[]
tmpnum = 0
for j in file1:
    temp1=j.split('	')[0]
    temp2=j.split('	')[1].rstrip('\n')
    if temp1 not in node:
        node.append(temp1)
        filetemp.write(temp1)
        filetemp.write('\n')
    if temp2 not in node:
        node.append(temp2)
        filetemp.write(temp2)
        filetemp.write('\n')
file1.close()
filetemp.close()

file1=open(str1)
#l是一个n*n的0矩阵，其中n是节点的个数
l = [([0] * len(node)) for x in range(len(node))]
#l中，将相互作用的地方变为1.下面这个for走完以后，就得到了0、1形式的临接矩阵了
for i in file1:
    temp1=i.split('	')[0]
    temp2=i.split('	')[1].rstrip('\n')
    q=0
    p=0
    for n in node:
        if n==temp1:
            a=node.index(n)
            q=1
            break
    for m in node:
        if m==temp2:
            b=node.index(m)
            p=1
            break
    if(q==1)and(p==1):
        l[a][b]=1
        l[b][a]=1
for i in range(len(node)):
    for j in range(len(node)-1):
        file2.write(str(l[i][j]))
        file2.write(' ')
    file2.write(str(l[i][len(node)-1]))
    file2.write('\n')
file2.close()

print ("create attributed network!")
#这里是生成0、1的属性矩阵。
go=[]
file=open("dataset/krogan2006core_go_information.txt")#"krogan2006extended_go_information.txt"
file4=open("dataset/ppin_noise/krogan2006core_go_information_temp.txt",'w')#"krogan2006extended_go_information_temp.txt",'w'
for i in file:
    node_name=i.split(' ',1)[0]
    node_go=i.split(' ',1)[1].rstrip('\n').rstrip(' ')
    file4.write(node_name)
    file4.write(' ')
    node_go=re.split(" |:",node_go)
    for j in node_go:
        if(j!='GO'):
            file4.write(j)
            file4.write(' ')
    file4.write('\n')
    for j in node_go:
        if j not in go:
            go.append(j)
file.close()
file4.close()
go.remove('GO')
go.sort()

file=open("dataset/krogan2006core_go_information_temp.txt")#"krogan2006extended_go_information_temp.txt"
gov=[]
for i in file:
    node_name = i.split(' ', 1)[0]
    node_go = i.split(' ', 1)[1].rstrip('\n').rstrip(' ')
    node_go=node_go.split(' ')
    one = {}
    one['node_name']=node_name
    one['node_go']=node_go
    gov.append(one)
file.close()

attr = [[0 for col in range(len(go))] for row in range(len(node))]
file3=open(str3,'w')
for i in node:
    for j in gov:
        if i==j['node_name']:
            a=node.index(i)
            for z in j['node_go']:
                for q in go:
                    if z==q:
                        b=go.index(q)
                        attr[a][b]=attr[a][b]+1
print ('success!')
for i in range(len(node)):
    for j in range(len(go)):
        file3.write(str(attr[i][j]))
        file3.write(' ')
    file3.write('\n')
file3.close()

