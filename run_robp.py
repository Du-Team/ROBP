#第一顺序主文件
import argparse   #命令行提供输入的
import numpy as np
import clustering_tools as ct   #自己写的文件
import border_tools as bt   #自己写的文件
import BorderPeel #自己写的文件

#这个是SK_Learn包
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import SpectralEmbedding #流形学习，拉普拉斯特征变换

# required是必须要输入
# action是默认值
# metavar是做注释用的
parser = argparse.ArgumentParser(description='Border-Peeling Clustering')
parser.add_argument('--input', type=str, metavar='<file path>',help='Path to comma separated input file', required=True)#输入文件
parser.add_argument('--output', type=str, metavar='<file path>',help='Path to output file', required=True)#输出文件
parser.add_argument("--no-labels", help="Specify that input file has no ground truth labels",action="store_true")#是否含有真的标签
parser.add_argument('--pca', type=int, metavar='<dimension>',help='Perform dimensionality reduction using PCA to the given dimension before running the clustering', required=False)#PCA降维
parser.add_argument('--spectral', type=int, metavar='<dimension>', help='Perform sepctral embdding to the given dimension before running the clustering (If comibined with PCA, PCA is performed first)', required=False)#拉普拉斯降维
args = parser.parse_args()#把输入的东西都拿到
output_file_path = args.output
input_file_path =  args.input
input_has_labels = not args.no_labels #输入--no-labels则为False 否则为True 就是无标签与有标签

pca_dim = args.pca #PCA的维数
spectral_dim = args.spectral #流形降维数

debug_output_dir = None #目前不知道什么东西，从数据观察的角度来看，应该是属于进行debug的方式
k=20 #KNN
C=3 #li的那个C
border_precentile = 0.1 #边界剥离百分比10%
mean_border_eps = 0.15 #终止条件那个0.15
max_iterations = 100 #最大迭代次数
stopping_precentile = 0.01 #停止百分比，目前不是很清楚，估计是数据很少就停止吧
data, labels = ct.read_data(input_file_path, has_labels=input_has_labels)

#判断数据量的大小
if len(data) < 1000:
    min_cluster_size = 10
else:
    min_cluster_size = 30

#把数据给基于流行
embeddings = data
#pca降维
if pca_dim is not None:
    if pca_dim >= len(embeddings[0]):
        print("PCA target dimension (%d) must be smaller than data dimension (%d)"%(pca_dim, len(embeddings[0])))
        exit(1)
    print("Performing PCA to %d dimensions"%pca_dim)
    pca = PCA(n_components=pca_dim)
    embeddings = pca.fit_transform(data)
#流行降维
if spectral_dim is not None:
    if spectral_dim >= len(embeddings[0]):
        print ("Spectral Embedding dimension (%d) must be smaller than data dimension (%d)"%(spectral_dim, len(embeddings[0])))
        exit(1)
    print("Performing Spectral Embedding to %d dimensions" % spectral_dim)
    se = SpectralEmbedding(n_components=spectral_dim)
    embeddings = se.fit_transform(data)

#降维好了后进行BP算法
print("Running Border-Peeling clustering on: %s"%input_file_path)
print("*"*60)
#这个先算那个第一次迭代的λ,初始化lamda
lambda_estimate = bt.estimate_lambda(embeddings, k)
#开始BP算法了，建立了一个bp类
bp = BorderPeel.BorderPeel(
                mean_border_eps=mean_border_eps #终止参数0.15
                ,max_iterations=max_iterations #最大迭代次数100
                ,k=k #K近邻20
                ,plot_debug_output_dir = None #暂时不知道是什么
                ,min_cluster_size = min_cluster_size #最小聚类大小
                ,dist_threshold = lambda_estimate #初始λ，也叫距离阈值
                ,convergence_constant = 0 #收敛常数，不知道是什么
                ,link_dist_expansion_factor = C #li的那个C就是3
                ,verbose = True #常态
                ,border_precentile = border_precentile #剥离百分比10%
                ,stopping_precentile = stopping_precentile #停止百分比1%不是很清楚，用来控制最大核心点数的
                )

#直接预测得到的类
clusters = bp.fit_predict(embeddings)
#数数有几个类，看看有没有-1在类里面如果有的话减掉1否则就是不减

clusters_count = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
print("*"*60)
print("Found %d clusters"%clusters_count)
print("*"*60)
#最后放起来
with open(output_file_path, "w") as handle:
    clustering_result=[]
    for c in clusters:
        clustering_result.append(c)
    handle.write(str(clustering_result))

print ("Saved cluster results to %s"%output_file_path)
print("*"*60)
#聚类的外部指标
if input_has_labels:
    print("ARI: %0.3f"%adjusted_rand_score(clusters, labels))
    print("AMI: %0.3f"%adjusted_mutual_info_score(clusters,labels))
