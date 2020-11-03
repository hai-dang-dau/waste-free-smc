import numpy as np
from programs.binary_regression.model import log_logit
import matplotlib.pyplot as plt
import pandas as pd

def generate(N, beta_0, beta_1, beta_2, force):
    """
    Generate a logistic regression scenario in dimension 2, with N points.
    The separating plan is beta_0 + beta_1 * x + beta_2 * y = 0.
    The true beta is force * [beta_0, beta_1, beta_2]
    """
    beta = force * np.array([beta_0, beta_1, beta_2])
    points = np.random.normal(scale=12, size=(N, 2))
    predictors = np.c_[np.ones(N), points]
    log_proba = log_logit((predictors @ beta[:, None])[:, 0])
    # noinspection PyUnresolvedReferences
    labels = (np.log(np.random.uniform(size=N)) < log_proba).astype(int)
    labels[labels == 0] = -1
    data = np.c_[labels, predictors]
    return data, beta

def visualize(data, beta):
    fig, ax = plt.subplots()
    df = pd.DataFrame({'label': data[:, 0], 'x': data[:, 2], 'y': data[:, 3]})
    df.plot.scatter(x='x', y='y', c='label', ax=ax, colormap='viridis')
    plot_line(beta=beta, ax=ax)
    plt.show()
    return

def plot_line(beta, ax):
    """
    Plot the line beta[0] + beta[1] * x + beta[2] * y = 0, while respecting the x and y limits of ax
    """
    xlim = ax.get_xlim()
    x = np.linspace(start=xlim[0], stop=xlim[1])
    y = -beta[0]/beta[2] - beta[1]/beta[2] * x
    ylim = ax.get_ylim()
    pos = np.where((y > ylim[0]) * (y < ylim[1]))
    x = x[pos]; y = y[pos]
    ax.plot(x, y)
    return

def run(beta_0, beta_1, beta_2, force):
    np.random.seed(2019)
    visualize(*generate(N=500, beta_0=beta_0, beta_1=beta_1, beta_2=beta_2, force=force))
    data_few, _ = generate(N=100, beta_0=beta_0, beta_1 = beta_1, beta_2=beta_2, force=force)
    data_some, _ = generate(N=1000, beta_0=beta_0, beta_1 = beta_1, beta_2=beta_2, force=force)
    data_lot, _ = generate(N=10000, beta_0=beta_0, beta_1 = beta_1, beta_2=beta_2, force=force)
    path = './programs/binary_regression/data/'
    np.save(path + 'sanity_few.npy', data_few)
    np.save(path + 'sanity_some.npy', data_some)
    np.save(path + 'sanity_lot.npy', data_lot)

if __name__ == '__main__':
    run(beta_0=-12, beta_1=3, beta_2=4, force=0.05)