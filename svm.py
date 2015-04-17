# global imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scopt
from matplotlib.mlab import PCA

# local imports

WHICH = 'Iris-setosa'


def load_data():
    data = open('data/iris.data')
    data = data.readlines()
    data = [i.split(',') for i in data]
    data = data[:len(data) - 1]
    data_container = np.zeros((len(data), len(data[0])-1))
    for i, item in enumerate(data):
        data_container[i] = [float(j) for j in item[:len(data[0]) - 1]]
    data_classifier = [1 if i[4] == WHICH + '\n' else -1 for i in data]
    return np.array(data_container), np.array(data_classifier)


def k(x):
    return np.dot(x.T, x)


def L(alpha, y, x):
    return - np.sum(alpha) + 1. / 2 * np.dot(alpha * y,
                                             np.dot(k(x), alpha * y))


def get_w(loc, alpha, y, x):
    w = np.sum([alpha[i] * y[i] * x[:, i] for i in loc], axis=0)
    return w

x, y = load_data()
x = x.T
alpha = np.zeros(len(y))

x = x[:, ::2]
y = y[::2]
alpha = alpha[::2]

cons = ({'type': 'eq',
         'fun': lambda a: np.dot(a, y),
         },
        {'type': 'ineq',
         'fun': lambda a: a,
         }
        )

res = scopt.minimize(L, alpha, args=(y, x), constraints=cons)
loc = np.where(abs(res['x']) > 1e-10)[0]
print 'idx', loc
print 'alphas', res['x'][loc]
w = get_w(loc, res['x'], y, x)
b = 1./len(loc) * np.sum([np.dot(w, x[:, i]) - y[i] for i in loc])
print 'w', w, 'b', b

print 'support vectors'
supx = []
supy = []
for i in loc:
    supx.append(x[:, i])
    supy.append(y[i])
    print np.dot(w, x[:, i]) - b

x, y = load_data()
# x = x.T
# cls = np.array([np.dot(w, x[:, i]) for i in xrange(len(y))]) - b
# srt = np.argsort(cls)

# for i in srt:
#     print cls[i], y[i]

pcax = PCA(x)

newdata = []

colordict = {
    # 'iris-setosa\n': 'r',
    # 'iris-virginica\n': 'm',
    # 'iris-versicolor\n': 'b',
    '1': 'r',
    '-1': 'b',
}

for i, xi in enumerate(x):
    newdata.append(pcax.project(xi))
    plt.plot(newdata[-1][0], newdata[-1][1], 'o',
             color=colordict[str(y[i])])

neww = pcax.project(w)
for i, xi in enumerate(np.array(supx)):
    newsupx = pcax.project(xi)
    print newsupx, np.dot(w, xi) - b
    plt.plot(newsupx[0], newsupx[1], 's',
             color=colordict[str(supy[i])])
    # plt.plot([newsupx[0], newsupx[0]+neww[0]],
    # [newsupx[1], newsupx[1]+neww[1]])


def f(x0):
    return (b - neww[0] * x0) / neww[1]

x0 = np.arange(-5, 5, 0.1)
plt.plot(x0, f(x0), 'k.')
plt.show()
