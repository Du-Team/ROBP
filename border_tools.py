#第三顺序功能函数
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import griddata
from clustering_tools import DebugPlotSession
from python_algorithms.basic import union_find
from time import time
import numpy as np
import pandas as pd
import copy
'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------
算RKNN的值函数
'''
#算rknn的值
def rknn_with_distance_transform(data, k, transform):
    #先数数有几个数据
    rows_count = len(data)
    #k取歌最小值，这里就是20了，因为另一个很可能比较大
    k = min(k , rows_count - 1)
    #弄一个数据点个数的0的数组
    rknn_values = np.zeros(rows_count)
    #先来k近邻
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    #获得距离和索引
    distances, indices = nbrs.kneighbors()

    for index, indRow, distRow in zip(range(len(indices)), indices,distances):
    #获得了基本形态如0 [  1 196 197 195 198] [1.25399362 2.76134025 2.93470612 3.18001572 3.55844067]
    #然后每次拿出后面两个列表的一对进行transform函数
        for i,d in zip(indRow, distRow):
            transform(rknn_values, index, i,d, indices, distances, k)
            #参数介绍一下，第一个是每个点的一个数组，第二个是数组中的索引就是点，第三个是i，第四个是d，第五个是索引，第六个是距离，第七个是k
    #返回了一个元组k和nbrs，就是一个数组里面装了bi和一个k近邻
    return (rknn_values, nbrs)
#算rknn的最核心函数
def exp_local_scaling_transform(rknnValues, first_index, second_index, dist, indices, distances, k):
    first_scale_index = k
    #这个判断语句是看看你实际上有几个点
    if len(distances[first_index]) <= first_scale_index:
        first_scale_index = len(distances[first_index]) - 1
    #本地σ就是那个exp上的本地σ
    local_sigma = distances[first_index][first_scale_index] #这个就是xj的的第k个数据点和本身之间的二范式了
    rknnValues[second_index] = rknnValues[second_index] + 1/(1+((dist * dist) /(local_sigma * local_sigma)))

'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------
单次边界剥离计算
'''
def border_peel_single(data, border_func, threshold_func, precentile=0.1, verbose=False):
    #这边给出的是前面比较重要，是bi，后面是k近邻的实例化模型
    border_values, nbrs = border_func(data)
    if border_values is None:
        return None,None,None
    # calculate the precentile of the border value..
    if precentile > 0:
        sorted = np.array(border_values)
        #排序
        sorted.sort()
        index_prcentile = int(len(border_values) * precentile)
        #这个threshold_value拿到的是10%的边界点的那个点
        threshold_value = sorted[index_prcentile]
        if verbose:
            print ("threshold value %0.3f for precentile: %0.3f" % (threshold_value, precentile))
        filter = border_values > threshold_value
    else:
        filter = threshold_func(border_values)
    #这返回了个判断True和False的数组,一个bi数组和一个knn的模型
    return filter, border_values, nbrs
'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------
函数用作判断是否有A1属性
'''
def mat_to_1d(arr):
    if hasattr(arr, 'A1'):
        return arr.A1
    return arr
'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
def evaluateLinkThresholds(data, filter, nbrs, dist_threshold):
    dataLength = len(data)
    xy = []
    Z = []

    distances, indices = nbrs.kneighbors()
    for index, indRow, distRow in zip(range(dataLength), indices,distances):
        if (not filter[index]):
            continue

        # look for the nearest neighbor whose isn't border
        for j,d in zip(indRow[1:], distRow[1:]):
            if (not filter[j]):
                xy.append(mat_to_1d(data[j]).tolist())
                Z.append(d)
                break

    # todo: using nearest method here in order to avoid getting nans..should also try
    # and see if using linear is better..
    if (len(Z) == 0):
        return None

    thresholds = griddata(np.matrix(xy), np.array(Z), data, method='nearest')
    #thresholds = griddata(np.matrix(xy), np.array(Z), data, method='linear')

    return thresholds
'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------
链接计算剩余核心点的新的li
'''
def update_link_thresholds(
        current_data, original_indices ,original_data,thresholds, dist_threshold, link_dist_expansion_factor, k=10
        ):
    # link_thresholds = update_link_thresholds(
    #     current_data  # 新的数据
    #     , original_indices  # 新的索引
    #     , original_data  # 最初的数据，就是完整的数据集吧
    #     , link_thresholds  # 前一次迭代的li数组
    #     , dist_threshold  # 应该是λ
    #     , link_dist_expansion_factor  # 大C就是3
    #     , k=k  # KNN的k
    # )
    # index the filters according to the original data indices:
    original_data_filter = np.zeros(len(original_data)).astype(int)
    original_data_filter[original_indices] = 1

    xy = original_data[(original_data_filter == 0)]
    Z = thresholds[(original_data_filter == 0)]
    # 这里获得的xy和Z分别是该轮迭代的边界点及其阈值

    knn = KNeighborsRegressor(k, weights="uniform")

    try:
        new_thresholds = knn.fit(xy, Z).predict(current_data)
        #做这个拟合回归就是找最近的k个点的均值
    except:
        print("failed to run kneighbours regressor")
        return thresholds

    for i, p, t in zip(range(len(current_data)), current_data, new_thresholds):
        #original_index = original_data_points_indices[tuple(p)]
        original_index = original_indices[i]

        if np.isnan(t):
            print ("threshold is nan")

        if np.isnan(t) or (t * link_dist_expansion_factor) > dist_threshold:
            #print "setting dist threshold"
            thresholds[original_index] = dist_threshold
        else:
            #print "setting threhold: %.2f"%(t * link_dist_expansion_factor)
            thresholds[original_index] = t * link_dist_expansion_factor

    return thresholds
'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------
这个是最核心函数
'''
#花费的时间构造类
class StopWatch:
    def __init__(self):
        self.time = time()
    def t(self, message):
        print("watch: %s: %0.4f" % (message, time() - self.time))

        # pass
        # self.time = time()
def border_peel(
                data          # 数据集
                ,border_func  # 边界剥离函数
                ,threshold_func  # 这里是None
                ,max_iterations=150  # 最大迭代次数
                ,min_iterations=3  # 最小迭代次数
                ,mean_border_eps=-1 # 终止条件参数
                ,plot_debug_output_dir=None # 画图
                ,min_cluster_size=3 #最小类数量
                ,dist_threshold=3 # 距离阈值li
                ,convergence_constant=0 #收敛参数
                ,link_dist_expansion_factor=3 #C
                ,k=10 #
                ,verbose=True
                ,precentile=0.1
                ,vis_data=None
                ,stopping_precentile=0 #终止阈值百分比
                ,should_merge_core_points=True #是否合并核心点集
                ,debug_marker_size=70 #调整标记大小
                ):
    """

    :type k: object
    """
    # 索引的元组散列
    # 创建时间类戳
    watch = StopWatch()

    #original_data_points_indices = {}
    data_length = len(data)
    cluster_uf = union_find.UF(data_length) #创建了一个并查集长度为数据个数
    original_indices = np.arange(data_length) #生成了一个长度数组，索引
    link_thresholds = np.ones(data_length) * dist_threshold  # 一开始判别的那个λ加入，就每个点都有的那个li
    #for d,i in zip(data,xrange(data_length)):
    #    original_data_points_indices[tuple(mat_to_1d(d))] = i

    #可视化的数据
    if vis_data is None:
        vis_data = data
    original_vis_data = vis_data    #最初的视觉数据
    current_vis_data = vis_data     #当前的视觉数据

    original_data = data    #最初的数据
    current_data  = data    #当前的数据
    data = None #把数据消除了

    # 1 if the point wasn't peeled yet, 0 if it was

    # original_data_filter  = np.ones(data_length)
    #这个也没用到，目前不知道干什么的

    # 这是画图的，先不管
    # plt_dbg_session = DebugPlotSession(plot_debug_output_dir, marker_size=debug_marker_size, line_width=1.0)
    # 放离群点等内容
    initial_core_points = []
    initial_core_points_original_indices = []
    #它又嵌了一层
    data_sets = [original_data]
    #我觉得可能就是从这个点来做计算所有和它距离的算法，这个可能是数据量很庞大
    nbrs = NearestNeighbors(n_neighbors=len(current_data)-1).fit(current_data)
    #拿到距离和索引
    nbrs_distances, nbrs_indices = nbrs.kneighbors()
    #最大核心点数？
    max_core_points = stopping_precentile * data_length
    #每次迭代的边界值的集合
    border_values_per_iteration = []

    if mean_border_eps > 0:
        mean_border_vals = []

    watch.t("initialization")
    #这边应该是开始迭代了吧
    '''
    ---------------迭代次数---------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------------------------------
    '''
    for t in range(max_iterations):
        start_time = time()
        #参数放入当前数据，范围KRNNbi数组的函数，第三个参数这边初始给的函数是None吧，剥离数量，这个verbose目前仍旧不知道是什么
        filter, border_values, nbrs = border_peel_single(
                                                        current_data
                                                        ,border_func
                                                        ,threshold_func
                                                        ,precentile=precentile
                                                        ,verbose=verbose
                                                        )
        # 这返回了个判断True和False的数组,一个bi数组和一个knn的模型
        watch.t("rknn")
        #这个就是去找被剥离的点呗，得到的应该是一个列表吧
        peeled_border_values = border_values[filter == False]
        #然后在列表里传递一个列表
        border_values_per_iteration.append(peeled_border_values)
        '''
        判断终止迭代
        '''
        if mean_border_eps > 0:
            mean_border_vals.append(np.mean(peeled_border_values))
            #这个判断是是否终止迭代的很重要
            if t >= min_iterations and len(mean_border_vals) > 2:
                ratio_diff = (mean_border_vals[-1] / mean_border_vals[-2]) - (mean_border_vals[-2] / mean_border_vals[-3])
                if verbose:
                    print ("mean border ratio difference: %0.3f"%(ratio_diff))
                if ratio_diff > mean_border_eps:
                    if verbose:
                        print ("mean border ratio is larger than set value, stopping peeling")
                    break

        #判断KNN是否有问题
        if nbrs is None:
            if verbose:
                print ("nbrs are none, breaking")
            break

        watch.t("mean borders")
        # filter out data points:开始过滤数据点了
        links = []

        #nbrs = NearestNeighbors(n_neighbors=len(current_data)-1).fit(current_data)
        #nbrs_distances, nbrs_indices = nbrs.kneighbors()

        original_data_filter = np.zeros(data_length).astype(int)
        #给的是数字，就是索引的数组
        original_indices_new = original_indices[filter]
        # print(original_indices_new)
        # print(len(original_indices_new))
        #将那些过滤点给那些大于10%阈值的点标为1，那些不大于的仍旧为0，核心点为1，边界点为0
        original_data_filter[original_indices_new] = 1
        # 这里每次迭代都要过滤的，存在疑问
        watch.t("nearset neighbors")
        #这一部分实际上是找最近的邻居
        for d, i, nn_inds, nn_dists in zip(current_data,range(len(current_data)), nbrs_indices, nbrs_distances):
            # skip non border points 跳过非边节点
            if filter[i]:
                continue

            # find the next neighbor we can link to 找到我们能够链接的点
            original_index = original_indices[i]
            # original_index = original_data_points_indices[tuple(d)]
            link_nig_index = -1 #链接点
            link_nig_dist = -1  #链接距离
            # make sure we exclude self point here.. 我们将自己给移出去
            # 根据li计算出来的东西，自己的li
            link_threshold = link_thresholds[original_index]
            #[  1 196 197 195 198] [1.25399362 2.76134025 2.93470612 3.18001572 3.55844067] 类似于这样子，不过这边更重的是所有点
            #还有就是要注意这是从小到大的
            for nig_index, nig_dist in zip(nn_inds, nn_dists):
                if nig_dist > link_threshold:
                    break

                #original_nig_index = original_indices[nig_index]
                #这个判断就是符合距离但不是核心点就再找
                if not original_data_filter[nig_index]:
                    continue

                #if filter[nig_index]:
                #这里就找到了
                link_nig_index = nig_index
                link_nig_dist = nig_dist
                break

            # do not link this point to any other point (but still remove it), consider it as noise instead for now
            # this will generally mean that this point is sorrounded by other border points
            #大于-1就说明找到了，把他链接在一起就可以了
            if link_nig_index > -1:
                links.append((i, link_nig_index))
                #original_link_nig_index = original_data_points_indices[tuple(current_data[link_nig_index])]
                #original_link_nig_index = original_indices[link_nig_index]
                original_link_nig_index = link_nig_index
                #做两部，就是把link链接的两个点放到链表和并查集里面
                cluster_uf.union(original_index, original_link_nig_index)
                #又做了一个改变就是把链接距离保存在了该点的li上
                link_thresholds[original_index] = link_nig_dist
            else: # leave it in a seperate cluster...
                initial_core_points.append(d)
                initial_core_points_original_indices.append(original_index)
                filter[i] = False
        # a = link_thresholds
        # calculate for next iterations

        for d, i, nn_inds, nn_dists in zip(current_data,range(len(current_data)), nbrs_indices, nbrs_distances):
            # skip non border points 跳过非边节点
            if filter[i]:
                continue

            # find the next neighbor we can link to 找到我们能够链接的点
            original_index = original_indices[i]
            for nig_index, nig_dist in zip(nn_inds, nn_dists):

                #original_nig_index = original_indices[nig_index]
                #这个判断就是符合距离但不是核心点就再找
                if original_data_filter[nig_index]:
                    continue
                tmp_a = nn_inds[0:k]
                tmp_ind  = np.where(original_indices == nig_index)
                tmp_b = nbrs_indices[tmp_ind,0:k]
                if np.intersect1d(tmp_a,tmp_b).size <= 0.5*k:
                    continue

                cluster_uf.union(original_index, nig_index)

        watch.t("association")
        #下面这个if判断应该是画图用的，先略
        # if (plot_debug_output_dir != None):
        #     original_data_filter = 2 * np.ones(len(original_data)).astype(int)
        #     for i,p,f in zip(range(len(current_data)), current_data,filter):
        #         #original_index = original_data_points_indices[tuple(p)]
        #         original_index = original_indices[i]
        #         original_data_filter[original_index] = f
        #
        #     plt_dbg_session.plot_and_save(original_vis_data, original_data_filter)

        # interpolate the threshold values for the next iteration:
        # 之前迭代的数据长度
        previous_iteration_data_length = len(current_data)
        # filter the data:
        current_data = current_data[filter]
        current_vis_data = current_vis_data[filter]
        #很重要的就是每次迭代的数据总量，下面四行都是迭代
        data_sets.append(current_data)
        original_indices = original_indices_new
        nbrs_indices = nbrs_indices[filter]
        nbrs_distances = nbrs_distances[filter]

        watch.t("filter")

        # calculate the link thresholds:
        # 链接计算新的数据点
        link_thresholds = update_link_thresholds(
                                                current_data    #新的数据
                                                ,original_indices   #新的索引
                                                ,original_data  #最初的数据，就是完整的数据集吧
                                                ,link_thresholds    #前一次迭代的li数组
                                                ,dist_threshold     #应该是λ
                                                ,link_dist_expansion_factor     #大C就是3
                                                ,k=k    #KNN的k
                                                 )
        # if False in a==link_thresholds:
        #     print(link_thresholds)
        watch.t("thresholds")

        if verbose:
            print ("iteration %d, peeled: %d, remaining data points: %d, number of sets: %d"\
                   %(t, abs(len(current_data) - previous_iteration_data_length),len(current_data), cluster_uf.count()))
        if abs(len(current_data) - previous_iteration_data_length) < convergence_constant:
            if verbose:
                print ("stopping peeling since difference between remaining data points and current is: %d"%(abs(len(current_data) - previous_iteration_data_length)))
            break

        if max_core_points > len(current_data):
            if verbose:
                print("number of core points is below the max threshold, stopping")
            break
    '''
    ---------------迭代次数---------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------------------------------
    '''
    '''
    ---------------聚类合并---------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------------------------------
    '''
    watch.t("before merge")
    clusters = np.ones(len(original_data)) * -1

    if verbose:
        print ("before merge: %d"%cluster_uf.count())
    print(initial_core_points_original_indices)
    core_points_merged = current_data.tolist() + initial_core_points

    original_core_points_indices = original_indices.tolist() + initial_core_points_original_indices

    core_points = np.ndarray(shape=(len(core_points_merged), len(core_points_merged[0])), buffer=np.array(core_points_merged))

    watch.t("before to associations map")

    #并查集合并成字典
    uf_map = uf_to_associations_map(cluster_uf, core_points, original_core_points_indices)

    watch.t("after associations map")
    non_merged_core_points = copy.deepcopy(core_points)

    if should_merge_core_points:
        merge_core_points(core_points, link_thresholds, original_core_points_indices, cluster_uf, verbose)


    watch.t("core points merge")

    if verbose:
        print ("after merge: %d"%cluster_uf.count())

    cluster_lists = union_find_to_lists(cluster_uf)

    cluster_index = 0

    for l in cluster_lists:
        if len(l) < min_cluster_size:
            continue

        for i in l:
            clusters[i] = cluster_index

        cluster_index += 1

    # core_clusters = -1.0 * np.ones(len(original_data)).astype(int)

    # if plot_debug_output_dir != None:
    #     for original_index in original_indices:
    #         core_clusters[original_index] = clusters[original_index]
    #
    #     # draw only core points clusters
    #     plt_dbg_session.plot_clusters_and_save(original_vis_data, core_clusters, noise_data_color = 'white')
    #
    #     # draw all of the clusters
    #     plt_dbg_session.plot_clusters_and_save(original_vis_data, clusters)

    watch.t("before return")

    return clusters, core_points, non_merged_core_points, data_sets, uf_map, link_thresholds, \
           border_values_per_iteration, original_indices
'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------
初始化lamda
'''
def estimate_lambda(data, k):
    '''
    算第一次的成对距离λ
    Parameters
    ----------
    data
    k

    Returns
    -------

    '''
    nbrs = NearestNeighbors(n_neighbors=k).fit(data) #k近邻 sklearn版本
    distances, indices = nbrs.kneighbors()

    all_dists = distances.flatten()
    return np.mean(all_dists) + np.std(all_dists)
'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------
并查集变为列表
'''
def union_find_to_lists(uf):
    list_lists = []
    reps_to_sets = {}

    for i in range(len(uf._id)):
        r = uf.find(i)
        if r not in reps_to_sets:
            reps_to_sets[r] = len(list_lists)
            list_lists.append([i])
        else:
            list_lists[reps_to_sets[r]].append(i)

    return list_lists

'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------
链接核心点
'''
def uf_to_associations_map(uf, core_points, original_indices):
    reps_items = {}
    reps_to_core = {}

    for original_index in original_indices:
        r = uf.find(original_index)
        reps_to_core[r] = original_index

    for i in range(len(uf._id)):
        r = uf.find(i)

        # 这不应该发生的
        if r not in reps_to_core:
            reps_to_core[r] = i

        k = reps_to_core[r]
        if  k not in reps_items:
            reps_items[k] = []
        reps_items[k].append(i)

    return reps_items
'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------
链接核心点
'''
def merge_core_points(core_points, link_thresholds, original_indices, cluster_sets, verbose=False):
    t = StopWatch()
    print(original_indices)
    try:
        nbrs = NearestNeighbors(n_neighbors=len(core_points) - 1).fit(core_points, core_points)
        distances, indices = nbrs.kneighbors()
    except Exception as err:
        if (verbose):
            print ("faiiled to find nearest neighbors for core points")
            print (err)
        return
    t.t("Core points - after nn")
    for original_index, ind_row, dist_row in zip(original_indices , indices, distances):
        #original_index = original_data_indices[tuple(p)]
        link_threshold = link_thresholds[original_index]
        for i,d in zip(ind_row[1:], dist_row[1:]):
            if d > link_threshold:
                break
            n_original_index = original_indices[i]
            cluster_sets.union(original_index, n_original_index)
    t.t("Core points - after merge")
'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
def border_peel_rknn_exp_transform_local(data, k, threshold, iterations, debug_output_dir=None,
                                         dist_threshold=3, link_dist_expansion_factor=3, precentile=0, verbose=True):
    border_func = lambda data: rknn_with_distance_transform(data, k, exp_local_scaling_transform)
    threshold_func = lambda value: value > threshold
    return border_peel(data, iterations, border_func, threshold_func,
                      plot_debug_output_dir=debug_output_dir, k=k, precentile=precentile,
                      dist_threshold=dist_threshold, link_dist_expansion_factor=link_dist_expansion_factor,
                      verbose=verbose)