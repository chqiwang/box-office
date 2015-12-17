# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 04:24:10 2015

@author: Sophia
"""
import pickle, getopt, sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale

def parse_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    with open('params.pickle') as f:
        params = pickle.load(f)
    
    m,n = len(lines),31+31+50+len(params['T'])+len(params['P'])+1+1+1
    X = np.zeros([m,n])
    names = []
    
    for i,line in enumerate(lines):
        units = line.split(',')
        units = [unit.decode('utf-8') for unit in units]
        X[i,0:31] = np.array([float(units[c]) for c in range(31)],float)
        X[i,31:62] = np.array([float(units[c]) for c in range(31,62)],float)
        img_path = units[62]
        img = plt.imread(img_path)
        img[:] = 0
        types = set(units[63].split(';'))
        for c,typ in enumerate(params['T']):
            if typ in types:
                X[i,112+c] = 1
        P = units[64]
        X[i,112+len(params['T'])+params['P'][P]] = 1
        X[i,112+len(params['T'])+len(params['P'])] = float(units[65])
        X[i,112+len(params['T'])+len(params['P'])+1] = float(units[67])
        X[i,112+len(params['T'])+len(params['P'])+2] = float(units[66]) - params['B']
        names.append(units[68].strip())
    
    X = minmax_scale(X[:,0:31])
    return X,names
    
def predict(file_name, result_file_name):
    try:
        X,names = parse_file(file_name)
    except Exception as e:
        print e
        print 'No File Found or Wrong Format!'
        with open(result_file_name,'w') as f:
            f.write(str(e)+'\n')
            f.write('No File Found or Wrong Format!')
        return
        
    with open('model.pickle') as f:
        model = pickle.load(f)
    Y = model.predict(X)
    
    with open('params.pickle') as f:
        params = pickle.load(f)
    Y = Y*params['Y']
    Y = np.asarray(Y/10,int)*10
    
    with open(result_file_name,'w') as f:
        for name,r in zip(names,Y):
            line = name.encode('utf-8')+','+str(r)
            f.write(line+'\n')
            print name.encode('utf-8')+' '+str(r)

if __name__ == '__main__':
    hint = __doc__ + '\n'\
          'usage\t: python predict.py -i [input file] -o [output file]\n'\
          'example\t: python predict.py -i input.csv -o result.csv.\n'\
          'Both input file and ouput file should be .csv format. About the details, see documents.\n'
    if len(sys.argv) == 1:
        print hint
        exit(1)
    input_file = None
    output_file = None
    opts, args = getopt.getopt(sys.argv[1:], "i:o:")
    for op, value in opts:
        if op == '-i':
            input_file = value
        elif op == '-o':
            output_file = value
    if input_file == None or output_file == None:
        print 'Error: Arguments are not satisfied.'
        print hint
        exit(1)
    predict(input_file, output_file)