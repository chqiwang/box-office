# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:49:41 2015

@author: pc
"""

import pickle,random,numpy,math
from pybrain.tools.shortcuts import *
from pybrain.datasets import *
from pybrain.supervised.trainers import *
from pybrain.structure import *

pkl_file1 = open('idx.pickle','rb');
pkl_file2 = open('movies.txt','rb');
 
data_baidu = pickle.load(pkl_file1);
data_movies = pickle.load(pkl_file2);
#print(len(data_movies));

list1 = list(data_baidu.keys());
#print list1[2];
#print data_baidu[list1[2]];
length = len(list1);
#print(length);

demension = 31;
data_train = numpy.zeros([599,demension]);
data_money = numpy.zeros(599);
total_money = 0;
n = 0;
max_zs = 0;
min_zs = 1000000000000;
max_news = 0;
min_news = 100000000000000;
for m in range (0,length):
    for i in range(0,len(data_movies)):   
        if(list1[m] == data_movies[i]['short_name']):
            total_money = data_movies[i]['total_money'];
            if (total_money < 10000001):
                for o in range(0,31):
                    data_train[n][o] = data_baidu[list1[m]][0][o];
                    if(max_zs < data_train[n][o]):
                        max_zs = data_train[n][o];
                    if(min_zs > data_train[n][o]):
                        min_zs = data_train[n][o]; 
                        '''
                for o in range(0,31):
                    data_train[n][o+31] = data_baidu[list1[m]][1][o];
                    if(max_news < data_train[n][o+31]):
                        max_news = data_train[n][o+31];
                    if(min_news > data_train[n][o+31]):
                        min_news = data_train[n][o+31];
                        '''
                data_money[n] = total_money;              
                n = n+1;
                break;
#print data_train[2];                
normalize_zs = max_zs-min_zs;
#normalize_news = max_news-min_news;
for n in range(0,599):
    for o in range(0,31):
        data_train[n][o] = (data_train[n][o]-min_zs)/normalize_zs;
        #data_train[n][o+31] = (data_train[n][o+31]-min_news)/normalize_news;
        
data_train = numpy.array(data_train);
#print data_train[2];


n = FeedForwardNetwork();
inLayer = LinearLayer(demension);
hiddenLayer = SigmoidLayer(200);
outLayer = LinearLayer(1);
 
n.addInputModule(inLayer);
n.addModule(hiddenLayer);
n.addOutputModule(outLayer);
 
in_to_hidden = FullConnection(inLayer, hiddenLayer);
hidden_to_out = FullConnection(hiddenLayer, outLayer);
 
n.addConnection(in_to_hidden);
n.addConnection(hidden_to_out);
 
n.sortModules();

DS = SupervisedDataSet(demension,1);
print
for i in range(0,len(data_train)):
    DS.addSample(data_train[i],data_money[i]);
X = DS['input'];
Y = DS['target'];

dataTrain, dataTest = DS.splitWithProportion(0.8);
#n = buildNetwork(62,500,1);
trainer = BackpropTrainer(n,dataTrain,verbose = False);
trainer.trainUntilConvergence(maxEpochs=500);
sum1 = 0;
for i in range(0,len(dataTest)):
    pre = n.activate(dataTest[i]);
    sum1 = sum1 + (abs(data_money[i]-pre))/data_money[i];
print sum1/len(dataTest);

