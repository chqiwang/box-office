import numpy as np
import scipy as sp
from itertools import *
import cPickle
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Extract actors
with open('../movies.txt') as f:
    M = cPickle.load(f)
for m in M:
    t = time.strptime(m['date'], "%Y-%m-%d")
    m['date'] = (t.tm_year, t.tm_mon)
M.sort(key = lambda x: x['total_money'], reverse = True)
M = M[2:]
M.sort(key = lambda x: x['date'])
names = [m['short_name'].encode('utf8') for m in M]
moneys = np.array([m['total_money'] for m in M])
actors = [m['actors'] for m in M]
dates = [m['date'] for m in M]
with open('names.pkl', 'w+') as f:
    cPickle.dump(names, f)
with open('moneys.pkl', 'w+') as f:
    cPickle.dump(moneys, f)
with open('actors.pkl', 'w+') as f:
    cPickle.dump(actors, f)
with open('dates.pkl', 'w+') as f:
    cPickle.dump(dates, f)

# Build actor dict
with open('actors.pkl') as f:
    A = cPickle.load(f)
actor_set = set(reduce(lambda l1, l2: l1 + l2, A))
actor_count = len(actor_set)
oneshot = lambda i: np.array([0]*i + [1] + [0]*(actor_count - 1 - i))
actor_dict = {}
for i, a in enumerate(actor_set):
    actor_dict[a] = oneshot(i)
with open('actor_dict.pkl', 'w+') as f:
    cPickle.dump(actor_dict, f)

# Vectorize
with open('actor_dict.pkl') as f:
    actor_dict = cPickle.load(f)
with open('actors.pkl') as f:
    A = cPickle.load(f)
def weighted_add(v):
    if len(v) == 1:
        weight = [1]
    elif len(v) == 2:
        weight = [0.7, 0.3]
    elif len(v) == 3:
        weight = [0.6, 0.3, 0.1]
    elif len(v) == 4:
        weight = [0.5, 0.3, 0.1, 0.1]
    elif len(v) == 5:
        weight = [0.5, 0.2, 0.1, 0.1, 0.1]
    return np.dot(weight, v)
actors_vector = np.array([weighted_add(np.array([actor_dict[a] for a in l])) for l in A])
with open('actors_vector.pkl', 'w+') as f:
    cPickle.dump(actors_vector, f)

# PCA
with open('actors_vector.pkl') as f:
    X = cPickle.load(f)
from sklearn.decomposition import PCA
pca = PCA(n_components = 605)
pca.fit(X)
X = pca.transform(X)
with open('actors_vector_rd.pkl', 'w+') as f:
    cPickle.dump(X, f)
print pca.explained_variance_ratio_

Show PCA results
with open('vectors_rd.pkl') as f:
    X = cPickle.load(f)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:,0], X[:,1], X[:,601])
plt.show()

# Normalize money by year
# B = lambda year: year * 451.040829 - 895232.310
# with open('moneys.pkl') as f:
#     M = cPickle.load(f)
# with open('dates.pkl') as f:
#     D = cPickle.load(f)
# for i in xrange(len(D)):
#     M[i] *= B(2000) / B(D[i][0])
# with open('moneys_normalized.pkl', 'w+') as f:
#     cPickle.dump(M, f)

# Show money distribution
# with open('moneys.pkl') as f:
#     M = cPickle.load(f)
# M.sort()
# l = M.shape[0]
# for i in range(1, 6):
#     print M[int(l * i / 6.0)]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xscale('log')
# ax.hist(M, 200)
# plt.savefig('money_distribution.png')

# Set money label
def get_label(money):
    if money < 2.8 * 10**3:
        return 0
    elif money < 4 * 10**3:
        return 1
    elif money < 7 * 10**3:
        return 2
    elif money < 12 * 10**3:
        return 3
    elif money < 22 * 10**3:
        return 4
    else:
        return 5
    # if money < 10**4:
    #     return 0
    # else:
    #     return 1
# with open('moneys_normalized.pkl') as f:
#     M = cPickle.load(f)
# money_labels = np.array([get_label(m) for m in M])
# plt.figure()
# plt.hist(money_labels, 6)
# plt.show()
# with open('moneys_labels.pkl', 'w+') as f:
#     cPickle.dump(money_labels, f)

# NN
with open('actors_vector_rd.pkl') as f:
    X = cPickle.load(f)
with open('moneys_labels.pkl') as f:
    Y = cPickle.load(f)
X0, Y0 = X, Y
X = X[:,:50]
oneshot = lambda y: [0]*y + [1] + [0]*(5-y)
Y = np.array([oneshot(y) for y in Y])
test_num = 100
X_train = X[:-test_num]
Y_train = Y[:-test_num]
X_test = X[-test_num:]
Y_test = Y[-test_num:]
import nn
nn.lamd = 0.5
nn.iterations = 10**3
nn.input_len = len(X[0])
nn.output_len = 6
nn.hidden_depth = 0
nn.hidden_unit_numbers = [3] * nn.hidden_depth
nn.hidden_activations = [nn.tanh] * nn.hidden_depth
nn.hidden_activations_d = [nn.d_tanh] * nn.hidden_depth
nn.output_activations = nn.sigmod
nn.output_activations_d = nn.d_sigmod
predictor, objvalues = nn.NN(X_train, Y_train)
print nn.error_rate(X_train, Y_train, predictor)
print nn.error_rate(X_test, Y_test, predictor)
for x, y in izip(X_test, Y0[-test_num:]):
    print y, predictor(x)
