#第二顺序类文件
import border_tools as bt
from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin


class BorderPeel(BaseEstimator,ClusterMixin):
    """Perform DBSCAN clustering from vector array or distance matrix.
    TODO: Fill out doc
    BorderPeel - Border peel based clustering
    Read more in the :ref:`User Guide <BorderPeel>`.
    Parameters
    ----------
    TODO: Fill out parameters..
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    Attributes
    ----------
    core_sample_indices_ : array, shape = [n_core_samples]
        Indices of core samples.
    Notes
    -----
    References
    ----------
    """

    # 初始化
    def __init__(self
                 ,method = "exp_local_scaling"
                 ,max_iterations = 150
                 ,mean_border_eps = -1 #条件终止参数
                 ,k=20
                 ,plot_debug_output_dir = None #画图参数
                 ,min_cluster_size = 3
                 ,dist_threshold = 3 #li
                 ,convergence_constant = 0 #收敛常数
                 ,link_dist_expansion_factor = 3 # C
                 ,verbose = True #常态 True
                 ,border_precentile = 0.1
                 ,stopping_precentile = 0
                 ,merge_core_points = True
                 ,debug_marker_size = 70
                 ):

        self.method = method
        self.k = k
        self.max_iterations = max_iterations
        self.plot_debug_output_dir = plot_debug_output_dir
        self.min_cluster_size = min_cluster_size
        self.dist_threshold = dist_threshold
        self.convergence_constant = convergence_constant
        self.link_dist_expansion_factor = link_dist_expansion_factor
        self.verbose = verbose
        self.border_precentile = border_precentile
        self.stopping_precentile = stopping_precentile
        self.merge_core_points = merge_core_points
        self.mean_border_eps = mean_border_eps
        self.debug_marker_size = debug_marker_size

        # out fields:输出参数
        self.labels_ = None #标签集
        self.core_points = None #核心点
        self.core_points_indices = None #核心点索引
        self.non_merged_core_points = None #没有合并的核心点
        self.data_sets_by_iterations = None #迭代的数据集
        self.associations = None #链接
        self.link_thresholds = None #连接阈值
        self.border_values_per_iteration = None #每次迭代的边界值

    #fit 函数
    def fit(self, X, X_plot_projection = None):
        """Perform BorderPeel clustering from features
        Parameters
        ----------
        X : array of features (TODO: make it work with sparse arrays)
        X_projected : A projection of the data to 2D used for plotting the graph during the cluster process
        """

        if (self.method == "exp_local_scaling"):
            #构造一个函数，函数的功能就是计算rknn，这是一个可以提速的地方，关注border_func函数
            border_func = lambda data: bt.rknn_with_distance_transform(data, self.k, bt.exp_local_scaling_transform)
            #threshold_func = lambda value: value > self.threshold

        result = bt.border_peel(X  #处理的数据
                            ,border_func   #bi函数
                            ,None   #也是一个函数，具体没看懂
                            ,max_iterations=self.max_iterations #最大迭代次数
                            ,mean_border_eps=self.mean_border_eps   #终止条件参数
                            ,plot_debug_output_dir=self.plot_debug_output_dir   #画图的吧
                            ,k=self.k #KNN的k，目前是20
                            ,precentile=self.border_precentile  #剥离百分比
                            ,dist_threshold=self.dist_threshold #距离阈值lamda
                            ,link_dist_expansion_factor=self.link_dist_expansion_factor #li的那个C就是3
                            ,verbose=self.verbose   #True常态
                            ,vis_data=X_plot_projection #在聚类过程中用于绘制图形的二维投影,None吧
                            ,min_cluster_size=self.min_cluster_size #最小聚类大小
                            ,stopping_precentile=self.stopping_precentile   #停止百分比
                            ,should_merge_core_points=self.merge_core_points   #合并的核心点集
                            ,debug_marker_size=self.debug_marker_size #调试标记大小 70
                                )

        #接收result传出来的数据
        self.labels_, self.core_points, self.non_merged_core_points, \
        self.data_sets_by_iterations, self.associations, self.link_thresholds, \
        self.border_values_per_iteration, self.core_points_indices = result

        return self

    #fit_predict 函数
    def fit_predict(self, X, X_plot_projection = None):
        """Performs BorderPeel clustering clustering on X and returns cluster labels.
        Parameters
        ----------
        X : array of features (TODO: make it work with sparse arrays)
        X_projected : A projection of the data to 2D used for plotting the graph during the cluster process
        Returns
        -------
        y : ndarray, shape (n_samples,)
            cluster labels
        """

        self.fit(X, X_plot_projection=X_plot_projection)
        #返回聚类标签值
        return self.labels_
