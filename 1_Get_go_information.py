#此文件用于得到每个节点对应的属性信息
#file2 file3分别打开节点文件和最后想要的节点+属性文件
file2=open("dataset/krogan2006core_node.txt")#krogan2006core_node.txt
file3=open("dataset/krogan2006core_go_information.txt","w")#"krogan2006core_go_information.txt","w"

#file1是GO属性文件 打开它之后，为匹配file2中的节点，为其生成对应的属性
for i in file2:
    i=i.rstrip('\n')
    print (i)
    file3.write(i)
    file3.write(' ')
    file1 = open("dataset/go_slim_mapping.tab.txt")
    for j in file1:
        node_name = j.split('	')[0]
        go_tag=j.split('	')[3]
        go = j.split('	')[5]
        if node_name==i:
            if go_tag=="P" or go_tag=='F' :
                if go!='' :
                    file3.write(go)
                    file3.write(' ')
    file1.close()
    file3.write('\n')

file2.close()
file3.close()