# do hierachical clustering of countries based on their time-series of incoming international seat capacity
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
import seaborn as sns
from dynamicTreeCut import cutreeHybrid
# custom imports
from ps_funks import hotPlot as hp
from ps_funks import juteUtils as jut

_cluster_methods = ['single', 'complete', 'average', 'ward', 'centroid', 'weighted', 'median']


class Clusterer:
    '''
    INPUT:
        method: str='ward' # method for distance between clusters
        method_fcluster: str='fcluster', # method to form flat clusters
        t: str='fcluster', # threshold to form flat clusters
        distance_metric: str='correlation' # 'correlation' or 'euclidean'
    '''
    def __init__(self, data: pd.DataFrame | list[pd.DataFrame],
                 method: str='ward', # method for distance between clusters
                 method_fcluster: str='fcluster', # method to form flat clusters
                 t: str='fcluster', # threshold to form flat clusters
                 column_name: str='airports',
                 distance_metric: str='correlation',
                 data_is_distance_matrix: bool=False
                 ):

        self.distance_metric = distance_metric
        if data_is_distance_matrix:
            self.df_dist = data
        else:
            self.df_dist = self.calculate_distance(data, distance_metric)
        self.df_dist.columns.name = column_name
        self.df_dist.index.name = ''
        self.dist_condensed = squareform(self.df_dist, checks=False)
        # parameters of the class
        self.method = 'ward'
        self.method_fcluster = 'fcluster'
        self.t = None

    def set_method(self, method: str):
        self.method = method

    def set_method_fcluster(self, method: str):
        valid_methods_fcluster = ['fcluster', 'cutreeHybrid']
        assert method in valid_methods_fcluster, f"Invalid method. Choose one of {valid_methods_fcluster}."
        self.method_fcluster = method

    def set_distance_threshold(self, t: float):
        self.t = t

    def warn_set_threshold(self):
        if self.t is None:
            return Warning("Distance threshold 't' is not set. Call 'set_distance_threshold(t)' to set it before plotting clusters.")
        pass

    def calculate_distance(self, data, metric='correlation'):
        if metric == 'correlation':
            return self.distance_from_correlation(data)
        elif metric == 'euclidean':
            return self.distance_from_euclidean(data)
        else:
            raise ValueError(f"Invalid metric. Choose 'correlation' or 'euclidean', got {metric}")

    def distance_from_euclidean(self, data):
        if isinstance(data, pd.DataFrame):
            from scipy.spatial.distance import pdist, squareform
            dist_condensed = pdist(data.T, metric='euclidean')
            dist_matrix = squareform(dist_condensed)
            return pd.DataFrame(dist_matrix, index=data.columns, columns=data.columns)
        elif isinstance(data, list):
            # Calculate Euclidean distance for each DataFrame, then combine
            distances = []
            for df in data:
                from scipy.spatial.distance import pdist, squareform
                dist_condensed = pdist(df.T, metric='euclidean')
                distances.append(squareform(dist_condensed)**2)
            # Combine distances (L2 norm across datasets)
            df_dist = pd.DataFrame(np.sqrt(np.sum(distances, axis=0)),
                                   index=data[0].columns, columns=data[0].columns)
            return df_dist
        else:
            raise ValueError("Data must be a DataFrame or a list of DataFrames")

    def distance_from_correlation(self, data):
        if isinstance(data, pd.DataFrame):
            return 1 - data.corr()
        elif isinstance(data, list):
            data =  [1 - df.corr() for df in data]
            data_square = np.array([df.values**2 for df in data])
            df_dist = pd.DataFrame(np.sqrt(np.sum(data_square, axis=0)),
                                   index=data[0].columns, columns=data[0].columns)
            return df_dist
        else:
            raise ValueError("Data must be a DataFrame or a list of DataFrames")

    def linkage(self, method: str | None=None):
        # Convert full distance matrix to condensed form
        if method is None:
            method = self.method
        Z = linkage(self.dist_condensed, method=method, optimal_ordering=True)
        return Z

    def fcluster(self, criterion: str='distance'):
        Z = self.linkage()
        self.warn_set_threshold()
        clusters = None
        if self.method_fcluster == 'fcluster':
            clusters = fcluster(Z, t=self.t, criterion=criterion)
        elif self.method_fcluster == 'cutreeHybrid':
            clusters = cutreeHybrid(Z, self.dist_condensed, cutHeight=self.t)['labels']
        clusters = np.array(clusters)
        clusters -= clusters.min()  # make cluster ids start from 0
        return clusters

    def get_elbow_data(self, max_clusters: int):
        elbow_data = {}
        for method in _cluster_methods:
            # create the clustering based on the distance matrix
            Z = self.linkage(method=method)
            last = Z[-max_clusters:, 2]
            last_rev = 1 * last[::-1]
            elbow_data[method] = last_rev
        return elbow_data

    def elbow_plot(self, max_clusters: int):
        elbow_data = self.get_elbow_data(max_clusters)
        f, axs = hp.subplots(len(elbow_data), sharex=True, size=0.3)
        idxs = np.arange(1, max_clusters+1)
        for ax, (method, d) in zip(axs, elbow_data.items()):
            # create the clustering based on the distance matrix
            ax.plot(idxs, d, linewidth=1, alpha=0.5)
            ax.scatter(idxs, d, s=5)
            hp.annotate_points_by_x_as_int(ax, idxs[:4], d[:4])
        ax.set(xlabel='Number of clusters above $d_c$', ylabel=r'Distance threshold $d_c$')
        hp.abc_plotLabels([0.3, 0.9], axs, abc=list(elbow_data.keys()), fontsize=8)
        f.suptitle('Distance $d_c$ where number of cluster $N_c$ increases', y=-0.03, fontsize=10)
        return f, axs, elbow_data

    def plot_elbow_relative_difference(self, max_clusters: int):
        elbow_data = self.get_elbow_data(max_clusters)
        f, axs = hp.subplots(len(elbow_data), sharex=True, size=0.3)
        # compute the relative difference
        elbow_data_rel_diff = {k: np.diff(d) / d[:-1]
                               for k, d in elbow_data.items()}
        for ax, (method, d) in zip(axs, elbow_data_rel_diff.items()):
            idxs = np.arange(2, len(d) + 2)
            ax.plot(idxs, d, linewidth=1, alpha=0.5)
            ax.scatter(idxs, d, s=5)
        ax.set(xlabel='Number of clusters', ylabel='Relative difference of distance threshold')
        hp.abc_plotLabels([0.3, 0.9], axs, abc=list(elbow_data.keys()), fontsize=8)
        return f, axs, elbow_data_rel_diff

    def get_second_derivative_elbow_data(self, max_clusters):
        elbow_data = self.get_elbow_data(max_clusters)
        # compute the second derivative
        elbow_data_second_deriv = {k: np.diff(d, n=2)
                                   for k, d in elbow_data.items()}
        return elbow_data_second_deriv

    def plot_elbow_second_derivative(self, max_clusters: int):
        elbow_data_second_deriv = self.get_second_derivative_elbow_data(max_clusters)
        f, axs = hp.subplots(len(elbow_data_second_deriv), sharex=True, size=0.4)
        # compute the second derivative
        for ax, (method, d) in zip(axs, elbow_data_second_deriv.items()):
            idxs = np.arange(2, len(d) + 2)
            ax.plot(idxs, d, linewidth=1, alpha=0.5)
            ax.scatter(idxs, d, s=5)
            hp.annotate_points_by_x_as_int(ax, idxs[:4], d[:4])
            # make horizontal line
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        axs[3].set(ylabel=r'Second derivative of distance threshold ($\partial^2 d_c$)')
        axs[-1].set(xlabel='Number of clusters ($N_c$) above $d_c$\n where change happens')
        f.suptitle(r'Second derivative of $d_c(N_c)$. Pick the $N_c$ where $\partial^2 d_c$ is max', y=-0.03, fontsize=10)
        hp.abc_plotLabels([0.3, 0.9], axs, abc=list(elbow_data_second_deriv.keys()), fontsize=8)
        return f, axs, elbow_data_second_deriv

    def set_distance_threshold_for_Ncluster(self, N: int):
        elbow_data = self.get_elbow_data(N+1)[self.method]
        self.t = np.mean(elbow_data[N-2:N])

    def plot_dendrogram(self):
        self.warn_set_threshold()
        df = self.df_dist
        Z = self.linkage(method=self.method)
        # plot the dendrogram
        f, ax = hp.subplots(1, aspect=0.2, size=2)
        dendrogram(Z, labels=df.columns, ax=ax, leaf_rotation=90, color_threshold=self.t)
        ax.set(ylabel='Distance')
        # change the fontsize of the xticklabels
        ax.tick_params(axis='x', labelsize=8)
        return f, ax

    def plot_hierarchical_distance_matrix(self,
                                          show_ticks: bool=False,
                                          figsize: list[float]=[8, 8]):
        m_dist = self.df_dist
        # sort the columns to create clusters in the heatmap
        linkage_matrix = self.linkage(method=self.method)
        # make colors if threshold given
        colors = None
        if self.t is not None:
            colors = [f'C{c}' for c in self.fcluster()]
        # plot the clustered heatmap
        f = sns.clustermap(m_dist, row_linkage=linkage_matrix, col_linkage=linkage_matrix,
                           cmap='viridis', figsize=figsize,
                           cbar_kws={'label': (f'Distance: {self.method}' + r'($d_{ij}$)' +
                                               '\n' + r'$d_{ij}=1-\rho_{ij}$')},
                           col_colors=colors, row_colors=colors)
        # Colorbar font sizes
        fontsize = figsize[0]*1.5
        cb = f.ax_cbar
        cb.set_ylabel(cb.get_ylabel(), fontsize=fontsize)
        cb.tick_params(labelsize=fontsize)
        if not show_ticks:
            f.ax_heatmap.set_xticks([])
            f.ax_heatmap.set_yticks([])
            f.ax_heatmap.set_xlabel(f.ax_heatmap.get_xlabel(), fontsize=fontsize)
        return f

    def column_to_cluster(self, as_df: bool=True):
        clusters = self.fcluster()
        cols = self.df_dist.columns
        # make dictionary with keys=cols, values=cluster
        cluster_dict = {col: int(cluster) for col, cluster in zip(cols, clusters)}
        if as_df:
            return pd.DataFrame.from_dict(cluster_dict, orient='index', columns=['cluster'])
        return cluster_dict

    def plot_country_cluster_worldmap(self):
        # visualize the clusters on the world map
        world = jut.gpd_get_world()
        df_cluster = self.column_to_cluster()
        world = world.merge(df_cluster, how='left', left_on='iso_a3', right_index=True)
        # drop nans
        world = world.dropna(subset=['cluster'])
        # change cluster-id to color
        world['cluster'] = world['cluster'].replace(
            {i: f'C{i}' for i in np.arange(df_cluster.max().values[0]+1)})
        # use a discrete colormap to plot the clusters
        f, ax = hp.subplots(1, aspect=1.5, size=2)
        world.plot(color=world['cluster'],
                   legend=False, ax=ax)
        world.boundary.plot(ax=ax, color='w', linewidth=0.3, alpha=0.3)
        ax.axis('off')
        ax.set_aspect('equal')
        f.tight_layout()
        return f, ax
