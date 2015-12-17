# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:16:59 2015

@author: Sophia
"""

import pickle
import datetime
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
from unbalanced_dataset import SMOTE
from sklearn.ensemble import AdaBoostClassifier

def useful_movie_set():
    with open('movies.txt') as f:
        movies = pickle.load(f)
    with open('idx.pickle') as f:
        idx = pickle.load(f)
        
    movie_set = []
    for movie in movies:
        date = datetime.datetime.strptime(movie['date'],'%Y-%m-%d').date()
        if date <= datetime.date(2011,2,1):
            continue
        
        short_name = movie['short_name']
        if short_name == u'蝙蝠侠' or short_name == u'功夫':
            continue
        
        (search_index,news_index) = idx[short_name]
        if max(search_index) == 0 and max(news_index) == 0:
            continue
        
        movie_useful = {}
        movie_useful['name'] = short_name
        movie_useful['search_idx'] = search_index
        movie_useful['news_idx'] = news_index
        movie_useful['date'] = date
        movie_useful['will_watch'] = movie['will_watch']
        movie_useful['total_money'] = movie['total_money']
        movie_useful['types'] = movie['types']
        movie_useful['product_type'] = movie['product_type']
        movie_useful['length'] = movie['length']
        movie_set.append(movie_useful)
    
    with open('useful_info.pickle','w') as f:
        pickle.dump(movie_set,f)

def split():
    with open('useful_info.pickle') as f:
        movies = pickle.load(f)
        
    n = len(movies)
    idx = list(range(n))
    random.shuffle(idx)
    
    sp = int(n*0.85)
    train_set = [movies[idx[i]] for i in range(sp)]
    test_set = [movies[idx[i]] for i in range(sp,n)]
    
    with open('train_test.pickle','w') as f:
        pickle.dump((train_set,test_set),f)

def normalize_money_with_date():
    with open('train_test.pickle') as f:
        train_set,test_set = pickle.load(f)
    
    money = float(np.max([movie['total_money'] for movie in train_set]))
    year_money = np.array([[movie['date'].year,float(movie['total_money'])/money] for movie in train_set],float)
    
    year_mean = np.zeros([5,2])
    for y in range(5):
        money = year_money[year_money[:,0] == 2011+y,1]
        plt.scatter(y*np.ones(np.shape(money)),money)
        mean = np.mean(money)
        year_mean[y,:] = np.array([1+y,mean],float)
    
    regressor = LinearRegression()
    regressor.fit(year_mean[:,0:1],year_mean[:,1])
    a,b = regressor.coef_, regressor.intercept_
    with open('coef.pickle') as f:
        coef = pickle.load(f)
        coef['normalize_year'] = {'a':a,'b':b,'base':2010}
    with open('coef.pickle','w') as f:
        pickle.dump(coef,f)
    
    print a,b,regressor.score(year_mean[:,0:1],year_mean[:,1])
    plt.plot(year_mean[:,1])
    plt.savefig('year_money.png')

def normalize():
    with open('train_test.pickle') as f:
        train_set,test_set = pickle.load(f)
    
    coef = {}
    types = set()
    for movie in train_set:
        types.update(set(movie['types']))
    n = len(types)
    type_list = list(types)
    for movie in train_set+test_set:
        t_s = set(movie['types'])
        v = np.zeros([1,n],float)
        for i in range(n):
            if type_list[i] in t_s:
                v[0,i] = 1
        movie['types'] = v
    coef['Normal_types'] = type_list   
    
    p_type = set([movie['product_type'] for movie in train_set])
    dic = {t:i for i,t in enumerate(p_type)}
    for movie in train_set+test_set:
        v = np.zeros([1,len(p_type)],float)
        if movie['product_type'] in dic.keys():
            v[0,dic[movie['product_type']]] = 1
        movie['product_type'] = v
    coef['Normal_product'] = dic
    
    with open('train_test_norm.pickle','w') as f:
        pickle.dump((train_set,test_set),f)
    with open('coef.pickle','w') as f:
        pickle.dump(coef,f)

def train_with_search_index(X,Y,W):
    regressor = GradientBoostingRegressor(loss='lad',n_estimators=25,learning_rate=0.1)
    regressor.fit(X,Y,W)   
    return regressor
    
    """
    result = []
    kf = cross_validation.KFold(n=m, n_folds=10, shuffle=True,random_state=None)
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        W_train, W_test = W[train_index], W[test_index]
        regressor.fit(X_train,y_train,W_train)
        result.append(score(regressor,X_test,y_test))
    print result,np.mean(result),np.std(result)
    plt.plot(result)
    """
    
    """
    for loss in ['ls', 'lad', 'huber', 'quantile']:
        for n in [25,50,100,200,500]:
            for learning_rate in [1,0.5,0.1,0.05,0.01]:
                regressor = GradientBoostingRegressor(loss=loss,n_estimators=n,learning_rate =learning_rate)
                result = []
                kf = cross_validation.KFold(n=m, n_folds=10, shuffle=True,random_state=None)
                for train_index, test_index in kf:
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = Y[train_index], Y[test_index]
                    W_train, W_test = W[train_index], W[test_index]
                    regressor.fit(X_train,y_train,W_train)
                    result.append(score(regressor,X_test,y_test))
                print loss,n,learning_rate,':',np.mean(result),np.var(result)
    """

def score(regressor,X_test,y):    
    y_pred = np.abs(regressor.predict(X_test))
    diff = np.divide(np.abs(y_pred-y),y)
    
    f,ax = plt.subplots()
    ax.plot(y)
    ax.plot(y_pred)
    
    return np.mean(diff)

def calc_W(Y):
    W = 1.0/Y
    return W

def train_with_kmeans(X,Y,W):
    k = 2
    c = KMeans(k)
    y = c.fit_predict(np.reshape(Y,[len(Y),1]))
    
    label,slabel = 0,1
    m0,m1 = np.mean(Y[y == 0]),np.mean(Y[y == 1])
    if m0 > m1:
        label = 1
    idx = np.where(y == label)[0]
    Ys = Y[y == label]
    ys = c.fit_predict(np.reshape(Ys,[len(Ys),1]))
    m0,m1 = np.mean(Ys[ys == 0]),np.mean(Ys[ys == 1])
    if m0 > m1:
        slabel = 0
    y[idx[ys==slabel]] = abs(1-label)
    
    z,o = float(sum(y == 0)),float(sum(y == 1))
    if z > o:
        r = z/o
    else:
        r = o/z
    smote = SMOTE(ratio=r/2, kind='regular')
    XS_train, yy_train = smote.fit_transform(X,y)
    
    s = AdaBoostClassifier(n_estimators=300)
    s.fit(XS_train,yy_train)
    #y_test_pred = s.predict(X_test)
    
    X_trains = [X[y == i] for i in range(k)]
    Y_trains = [Y[y == i] for i in range(k)]
    #X_tests = [X_test[y_test_pred == i] for i in range(k)]
    #Y_tests = [Y_test[y_test_pred == i] for i in range(k)]
    W_trains = [W[y == i] for i in range(k)]
    
    regressors = []
    for i in range(k):
        regressor = LinearRegression()
        regressor.fit(X_trains[i],Y_trains[i],W_trains[i])
        regressors.append(regressor)
    return s,regressors

def ensemble():
    with open('train_test_norm.pickle') as f:
        train_set,test_set = pickle.load(f)
    m,n = len(train_set),len(train_set[0]['search_idx'])
    X = np.zeros([m,n])
    Y = np.zeros([m,])
    for i in range(m):
        movie = train_set[i]
        X[i,:] = np.asarray(movie['search_idx'],float)
        Y[i] = float(movie['total_money'])
    
    Y /= np.max(Y)
    X = minmax_scale(X)
    W = calc_W(Y)
    
    result = []
    kf = cross_validation.KFold(n=m, n_folds=10, shuffle=True,random_state=None)
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        W_train, W_test = W[train_index], W[test_index]
        
        boost = train_with_search_index(X_train,y_train,W_train)
        classifier,regressors = train_with_kmeans(X_train,y_train,W_train)
        
        result.append(score_en(boost,classifier,regressors,X_test,y_test))
        
def score_en(boost,classifier,regressors,X_test,y_test):
    X_boost = boost.predict(X_test)
    label = classifier.predict(X_test)
    y_pred = np.zeros([len(y_test),])
    
    coef = [[0.7,0.2],[0.6,0.4]]
    for i in range(len(regressors)):
        num = np.sum(label == i)
        X_regres = coef[i]*np.reshape(regressors[i].predict(X_test[label==i]),[num,1])
        X_p = np.hstack((np.reshape(X_boost[label==i],[num,1]),X_regres))    
        y_pred[label == i] = X_p[:,0]*coef[i][0] + X_p[:,1]*coef[i][1]
    
    y_pred = np.abs(y_pred)
    diff = np.divide(np.abs(y_pred-y_test),y_test)
    
    f,ax = plt.subplots()
    ax.plot(y_test)
    ax.plot(y_pred)
    
    return np.mean(diff)

#useful_movie_set()
#split()
#normalize_money_with_date()
#normalize()
#train_with_search_index()
#train_with_kmeans()
#ensemble()