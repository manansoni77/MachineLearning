from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.ticker as plticker
from matplotlib import pyplot
import pandas
import numpy
import math

colors = {0: 'red', 1: 'green', 2: 'blue'}


def sqrt_int(X: int):
    N = math.floor(math.sqrt(X))
    while bool(X % N):
        N -= 1
    M = X // N
    return int(M), int(N)


def reshape(arr):
    return numpy.array(arr).reshape(-1, 1)


def ordinal(arr):
    ord = OrdinalEncoder()
    return ord.fit_transform(arr)


def data_to_color(data, colors=colors):
    arr = OrdinalEncoder().fit_transform(data)
    arr = [colors[x[0]] for x in arr]
    return arr


class ClusteringModel:
    def __init__(self, data, input_features, output_labels):
        # initialise input and label
        self.data = data
        self.input_features = input_features
        self.output_labels = output_labels
        self.features = data[input_features]
        self.labels = data[output_labels]

    def train_kmeans(self, plot_file_name='pred.png', n_clusters=3, n_init=1, init='random'):
        # initialise kmeans
        self.kmeans = KMeans(n_clusters=n_clusters, init=init,
                             n_init=n_init, verbose=3, random_state=26)
        self.kmeans.fit(self.features, self.labels)

        # calculate score, and confusion matrix
        score = self.kmeans.score(self.features, self.labels)
        pred = self.kmeans.predict(self.features)
        confusion = confusion_matrix(ordinal(
            self.labels), ordinal(reshape(pred)))

        # plot data and labels,
        self.get_plots(reshape(pred), plot_file_name)

        return {"scores": score, "centers": self.kmeans.cluster_centers_, "params": self.kmeans.get_params(), "confusion": confusion}

    def get_plots(self, labels, name):
        cnt = 0
        nf = len(self.input_features)
        ncols, nrows = sqrt_int((nf*(nf-1))/2)
        fig, ax = pyplot.subplots(nrows=nrows, ncols=ncols)

        for i, feat1 in enumerate(self.input_features[:-1]):
            for feat2 in self.input_features[i+1:]:
                r = cnt // ncols
                c = cnt % ncols
                cnt += 1

                ax[r][c].scatter(data[feat1], data[feat2],
                                 c=data_to_color(labels))
                ax[r][c].set_xlabel(feat1)
                ax[r][c].set_ylabel(feat2)
                ax[r][c].xaxis.set_major_locator(plticker.MaxNLocator(5))
                ax[r][c].yaxis.set_major_locator(plticker.MaxNLocator(5))

        fig.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(name)
        return fig

    def predict(self, x):
        return self.kmeans.predict(x)


if __name__ == '__main__':
    data = pandas.read_csv('./iris.csv')
    features = data.columns
    clustering = ClusteringModel(data, features[1:5], features[5:6])
    print(clustering.train_kmeans())
    pred = clustering.predict(data[features[1:5]])
    plot = clustering.get_plots(reshape(pred), 'plot.png')
    plot.show()
