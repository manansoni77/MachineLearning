import pandas
import numpy
from sklearn.linear_model import LinearRegression


def reshape(arr):
    return numpy.array(arr).reshape(-1, 1)


class LinearModel:
    def __init__(self, x, y, fit_intercept=True, series=True):
        self.series = series

        # initialise x and y
        if self.series:
            x = reshape(x)
            y = reshape(y)

        # intialise regressor
        self.reg = LinearRegression(fit_intercept=fit_intercept)
        self.reg.fit(x, y)
        self.score = self.reg.score(x, y)

    def predict(self, x):
        if self.series:
            x = reshape(x)

        return self.reg.predict(x)


if __name__ == '__main__':
    data = pandas.read_csv('Experience-Salary.csv')
    features = data.columns

    x, y = data[features[0]], data[features[1]]
    model = LinearModel(x, y)
    print(model.score)
