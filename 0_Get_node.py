file1=open("dataset/krogan2006core.txt")
file2=open("dataset/krogan2006core_node.txt","w")

node=[]
for j in file1:
    temp1=j.split('	')[0]
    temp2=j.split('	')[1].rstrip('\n')
    if temp1 not in node:
        node.append(temp1)
        file2.write(temp1)
        file2.write('\n')
    if temp2 not in node:
        node.append(temp2)
        file2.write(temp2)
        file2.write('\n')
file1.close()
file2.close()