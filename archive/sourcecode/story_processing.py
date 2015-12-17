import numpy as np
import scipy as sp
from itertools import *
import cPickle
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Extract year, month, name, money and story
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
stories = [m['story'].encode('utf8') for m in M]
dates = [m['date'] for m in M]
plt.hist(dates[0], 9)
plt.show()
import jieba
def cut(s):
    W = jieba.cut(s)
    return ' '.join(W)
stories_cut = [cut(s) for s in stories]
with open('names.pkl', 'w+') as f:
    cPickle.dump(names, f)
with open('moneys.pkl', 'w+') as f:
    cPickle.dump(moneys, f)
with open('stories.pkl', 'w+') as f:
    cPickle.dump(stories, f)
with open('stories_cut.pkl', 'w+') as f:
    cPickle.dump(stories_cut, f)
with open('dates.pkl', 'w+') as f:
    cPickle.dump(dates, f)

with open('dates.pkl') as f:
    D = cPickle.load(f)
D = np.array(D)
plt.hist(D[:,0], 30)
plt.show()


# vectorize
from sklearn.feature_extraction.text import *
with open('stories_cut.pkl') as f:
    S = cPickle.load(f)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(S)
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)
with open('vectors.pkl', 'w+') as f:
    cPickle.dump(X, f)

# PCA
with open('vectors.pkl') as f:
    X = cPickle.load(f)
X = X.toarray()
from sklearn.decomposition import PCA
pca = PCA(n_components = 605)
pca.fit(X)
X = pca.transform(X)
with open('vectors_rd.pkl', 'w+') as f:
    cPickle.dump(X, f)
print pca.explained_variance_ratio_

# Show PCA results
# with open('vectors_rd.pkl') as f:
#     X = cPickle.load(f)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X[:,0], X[:,1], X[:,601])
# plt.show()

# Normalize money by year
B = lambda year: year * 451.040829 - 895232.310
with open('moneys.pkl') as f:
    M = cPickle.load(f)
with open('dates.pkl') as f:
    D = cPickle.load(f)
for i in xrange(len(D)):
    M[i] *= B(2000) / B(D[i][0])
with open('moneys_normalized.pkl', 'w+') as f:
    cPickle.dump(M, f)

# Show money distribution
with open('moneys.pkl') as f:
    M = cPickle.load(f)
M.sort()
l = M.shape[0]
for i in range(1, 6):
    print M[int(l * i / 6.0)]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.hist(M, 200)
plt.savefig('money_distribution.png')

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
    # return 1
with open('moneys_normalized.pkl') as f:
    M = cPickle.load(f)
money_labels = np.array([get_label(m) for m in M])
plt.figure()
plt.hist(money_labels, 6)
plt.show()
with open('moneys_labels.pkl', 'w+') as f:
    cPickle.dump(money_labels, f)

# Show low dimention distribution
with open('vectors_rd.pkl') as f:
    X = cPickle.load(f)
with open('moneys_labels.pkl') as f:
    Y = cPickle.load(f)
colors = ['r', 'g', 'b', 'c', 'y', 'm']
fig = plt.figure()
ax = Axes3D(fig)
for x, y in izip(X, Y):
    ax.scatter(x[0], x[1], x[2], color = colors[y])
plt.show()

# NN
with open('vectors_rd.pkl') as f:
    X = cPickle.load(f)
with open('moneys_labels.pkl') as f:
    Y = cPickle.load(f)
X0, Y0 = X, Y
X = X[:, :90]
oneshot = lambda y: [0]*y + [1] + [0]*(5-y)
Y = np.array([oneshot(y) for y in Y])
test_num = 100
X_train = X[:-test_num]
Y_train = Y[:-test_num]
X_test = X[-test_num:]
Y_test = Y[-test_num:]
import nn
nn.lamd = 0.2
nn.iterations = 10**2*2
nn.show_info = 100
nn.input_len = len(X[0])
nn.output_len = 6
nn.hidden_depth = 0
nn.hidden_unit_numbers = [8] * nn.hidden_depth
nn.hidden_activations = [nn.tanh] * nn.hidden_depth
nn.hidden_activations_d = [nn.d_tanh] * nn.hidden_depth
nn.output_activations = nn.sigmod
nn.output_activations_d = nn.d_sigmod
predictor, objvalues = nn.NN(X_train, Y_train)
print nn.error_rate(X_train, Y_train, predictor)
print nn.error_rate(X_test, Y_test, predictor)
for x, y in izip(X_test, Y0[-test_num:]):
    print y, predictor(x)