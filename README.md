运行步骤：
------------
0_Get_node.py: 得到相互作用网络里面的所有节点
1_Get_go_information.py: 得到每个节点的GO属性
2_Create_topological_biological_network.py: 创造出以“0/1”形式表示的邻接矩阵和属性矩阵
3_Node_embedding: 进行网络嵌入，得到每个蛋白质节点的低维向量表示
4_Get_cluster_score.py: 得到发生相互作用节点间的聚合分数
5_cluster_core_attachment.py: 聚类得到核心团、再添加附属蛋白质，得到结果
6_Compare_performance.py: 用于各种评价指标
------------
代码里面，也有相应的详细注释，可供参考。

