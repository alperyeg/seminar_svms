# global imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scopt
# from matplotlib.mlab import PCA


def princomp(A):
    '''
    Perform a principal component analysis on a given data matrix.

    Parameters
    ----------
    A : array_like
        A 1-D or 2-D array containing M variables and N observations.
        Each column of `A` represents a variable, and each row a single
        observation of those variables.

    Returns
    -------
    w : ndarray, shape (M,)
        The principal component eigenvalues (eigenvalues of the data
        covariance matrix) in decending order by magnitude.

    V : ndarray, shape (M, M)
        The normalized (unit "length") eigenvectors of the data
        covariance matrix (principal component coefficients), ordered
        such that the column ``V[:,i]`` is the eigenvector
        corresponding to the eigenvalue ``w[i]``.

    score : ndarray
        The projection of the data matrix on to the principal
        component space (principal component scores).
    '''

    # centralise the data
    # (subtract sample mean from each variable)
    X = A - np.mean(A, axis=0)

    # compute the covariance matrix from the centralised data
    # (using the dot product, as explained in the lecture)
    C = np.dot(X.T, X)/(X.shape[1]-1)

    # compute the eigenvectors of C
    [w,V] = np.linalg.eig(C)

    # get indexes in descending order of eigenvalue
    idx = np.argsort(-w)

    # resort eigenvalues and discard the imaginary part
    w = np.real(w[idx])

    # resort eigenvectors
    V = V[:,idx]

    # hint: the matrix `V` is the transformation matrix
    # we use to project the data into PCA space (below)

    # project data in PCA space and discard the imaginary part
    score = np.real(np.dot(X, V))

    return w, V, score

# local imports

WHICH = 'Iris-setosa'


def load_data():
    data = open('data/iris.data')
    data = data.readlines()
    data = [i.split(',') for i in data]
    data = data[:len(data) - 1]
    data_container = np.zeros((len(data), len(data[0]) - 1))
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

x -= np.mean(x, axis=0)

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
b = 1. / len(loc) * np.sum([np.dot(w, x[:, i]) - y[i] for i in loc])
print 'w', w, 'b', b

print 'support vectors'
supx = []
supy = []
for i in loc:
    supx.append(x[:, i])
    supy.append(y[i])
    print np.dot(w, x[:, i]) - b

x, y = load_data()
x -= np.mean(x, axis=0)

cls = np.array([np.dot(w, x.T[:, i]) for i in xrange(len(y))]) - b
srt = np.argsort(cls)

for i in srt:
    print (cls[i], y[i]),
print

# pcax = PCA(x, standardize=True)
w_eigen, V, score = princomp(x)

newdata = []

colordict = {
    # 'iris-setosa\n': 'r',
    # 'iris-virginica\n': 'm',
    # 'iris-versicolor\n': 'b',
    '1': 'r',
    '-1': 'b',
}

# for i, xi in enumerate(x):
#     plt.plot(xi[0], xi[1], 'o', color=colordict[str(y[i])])
# for i, xi in enumerate(np.array(supx)):
#     plt.plot(xi[0], xi[1], 's', color=colordict[str(supy[i])])
# plt.show()

for i, xi in enumerate(x):
    # newdata.append(pcax.project((xi + pcax.mu) * pcax.sigma))
    # newdata.append(pcax.project(xi))
    newdata.append(np.dot(V, xi))
    plt.plot(newdata[-1][2], newdata[-1][1], 'o',
             color=colordict[str(y[i])])

# neww = pcax.project(w)
for i, xi in enumerate(np.array(supx)):
    # newsupx = pcax.project(xi)
    newsupx = np.dot(V, xi)
    print newsupx, np.dot(w, xi) - b
    plt.plot(newsupx[2], newsupx[1], 's',
             color=colordict[str(supy[i])], markersize=8.)

# # project: Y = np.dot(self.Wt, self.center(x).T).T
# xpre = np.array([1, 2, 3, 4])
# xproj = pcax.project(xpre)
# print 'xproj', xproj
# print 'sigma/mu', pcax.sigma, pcax.mu
# print 'Wt', pcax.Wt
# xorig = np.dot(np.linalg.inv(pcax.Wt), xproj) * pcax.sigma + pcax.mu
# # print xproj, np.dot(pcax.Wt, (xpre-pcax.mu) / pcax.sigma)
# print xpre, xorig
# print 'Wtdot'
# print np.dot(pcax.Wt.T, pcax.Wt)

# W = np.linalg.inv(pcax.Wt).T
Vinv = np.linalg.inv(V)

# neww = np.dot(V, w)
print 'check', np.dot(Vinv, V)
print 'check', np.dot(V.T, V)
print np.dot(w, supx[0]) - b, np.dot(w, supx[1]) - b
wnew = np.dot(V, w)
print np.dot(wnew, np.dot(V, supx[0])) - b, np.dot(wnew, np.dot(V, supx[1])) - b

print supx


def f(x0):
    return 1. * (b - wnew[2] * x0) / wnew[1]


x0 = np.arange(-5, 5, 0.1)
plt.plot([-1., 1.], [f(-1.), f(1.)], 'k')
plt.plot([0, wnew[2]], [0, wnew[1]], 'g')
plt.xlim([-2., 2.])
plt.ylim([-2., 2.])
plt.show()
