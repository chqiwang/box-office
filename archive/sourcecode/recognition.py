# -*- coding: utf-8 -*-
"""
Created on Tue Dec 08 20:13:01 2015

@author: Sophia
"""

from matplotlib import pyplot as plt
from skimage.morphology import binary_closing
from skimage import measure
import numpy as np
import math
import pickle

###############################################################################
INDEX_GAP = 30
NEWS_GAP = 25

templates = [np.load('templates\\'+str(i)+'.npy') for i in range(10)]
tr,tc = (10,6)
###############################################################################

def split(name):
    img = plt.imread(name)
    
    img = img[205:495,255:980]
    
    binary = np.logical_and(img[:,:,2] > 0.8,img[:,:,0] < 0.5)
    binary = binary_closing(binary)
    binary_index = binary[:212,:]
    binary_news = binary[212:,:]
    
    img = img[:,600:]
    binary = img[:,:,2] < 0.9
    digit_index = binary[10:200,50:116]
    digit_news = binary[243:278,80:116]
    
    c = 30
    d = 12
    digit_index_list = [digit_index[i*c:i*c+d,:] for i in range(7)]
    
    digit_news_list = [digit_news[:11,:],digit_news[25:,:]]    
    
    return binary_index,binary_news,digit_index_list,digit_news_list

def __digit_bbox(img):
    label = measure.label(img,connectivity=2)
    label[img == 0] = 0
    
    props = measure.regionprops(label)
    bboxes = [prop.bbox for prop in props]

    bboxes = [box for box in bboxes if box[2]-box[0] > 5]    
    temp,bboxes = bboxes,[]
    gap = 7.
    for box in temp:
        n = int(math.ceil((box[3] - box[1])/gap-0.5))
        if n <= 1:
            bboxes.append(box)
        else:
            for i in range(n):
                bboxes.append((box[0],box[1]+i*int(gap),box[2],box[1]+(i+1)*int(gap)))
    
    bboxes = sorted(bboxes,key=lambda x:x[1])
    digits = [img[box[0]:box[2],box[1]:box[3]] for box in bboxes]    
    return digits

def __score(template,digits):
    dr,dc = np.shape(digits[0])
    n = abs(dc-tc)+1
    m = tr - dr + 1
    
    scores = []
    for i in range(n):
        for j in range(m):
            for digit in digits:
                if dc > tc:
                    r = np.logical_and(digit[:,i:i+tc],template[j:j+dr,:])
                    o = np.logical_or(digit[:,i:i+tc],template[j:j+dr,:])
                    score = float(np.sum(r))/float(np.sum(o))
                else:
                    r = np.logical_and(digit,template[j:j+dr,i:i+dc])
                    o = np.logical_or(digit,template[j:j+dr,i:i+dc])
                    score = float(np.sum(r))/float(np.sum(o))
                scores.append(score)
    return max(scores)

def __digit_recognize(digit):
    dr,dc = np.shape(digit)

    digits = []
    if dr > tr and (dr-tr)%2 == 0:
        n = (dr-tr)/2
        digits.append(digit[n:dr-n,:])
    elif dr > tr and (dr-tr)%2 == 1:
        n = (dr-tr)/2
        digits.append(digit[n:dr-n-1,:])
        digits.append(digit[n+1:dr-n,:])
    else:
        digits.append(digit)

    scores = [__score(template,digits) for template in templates]    
    return max(scores),np.argmax(scores)
    
def __digits_recognize(digits):
    s = '0'
    start = False
    for digit in __digit_bbox(digits):
        t,n = __digit_recognize(digit)
        if t >= 0.5 or start:
            start = True
            s = s + str(n)
    return int(s)

def __gap_fit(digits_list):
    nums = [__digits_recognize(digits) for digits in digits_list]
    if len(nums) > 2 and nums[0] <= nums[1] and nums[0] <= nums[2]:
        nums[0] = 2*nums[1] + nums[2]
    for i in range(1,len(nums)-1):
        if not (nums[i] < nums[i-1] and nums[i] > nums[i+1]):
            nums[i] = (nums[i-1]+nums[i+1])/2
    if len(nums) > 2 and nums[-1] > nums[-2] and nums[-1] > nums[-3]:
        nums[-1] = 2*nums[-2] - nums[-3]
    if len(nums) == 2 and nums[0] < nums[1]:
        nums[0] = 2*nums[1] - nums[0]
    
    gaps = [(nums[i-c]-nums[i])/c for c in range(1,len(nums)) for i in range(c,len(nums)) if nums[i-c]-nums[i] >= 0]
    gap = int(np.median(gaps))
    if gap <= 0:
        return (0,0)
        
    base = 10**int(math.log(gap,10))    
    gap = gap/base*base
    
    for i in range(len(nums)):
        if nums[i] < gap:
            m = (nums[i]+gap)/2
            gap = nums[i] = int(m)
            gap = gap/base*base
        
    bottoms = [int(nums[i]) - (len(nums)-i-1)*gap for i in range(len(nums))]
    minmum = int(min(nums))
    bottoms = [n for n in bottoms if n > 0 and n <= minmum and n >= gap]
    bottom = int(np.median(bottoms))
    
    return (gap,bottom-gap)

def __index(img):
    r,c = np.shape(img)
    start = 0
    while start < c & len(np.where(img[:,start])[0]) == 0:
        start += 1
    if start >= 10:
        start = 2
    
    end = c-1
    while end >= 0 and len(np.where(img[:,end])[0]) == 0:
        end -= 1
    if end < 710:
        end = min(c-1,717)   
    
    indexes = []
    for i in range(31):
        idx = int(min(start + i*((end-start)/30.),c-1))
        locs = np.where(img[:,idx])[0]
        if len(locs) == 0:
            indexes.append(0)
            continue
        
        loc = int(np.median(locs))
        indexes.append(r-loc+1)
    
    m = min(indexes)
    return [i-m for i in indexes]
        
def img2index(index,digit_list,vgap):
    gap,base = __gap_fit(digit_list)
    idx = __index(index)
    
    return [int(i/float(vgap)*gap+base) for i in idx]

def index2file():
    with open('movies.txt') as f:
        movies = pickle.load(f)
        
    idxs = {}
    for movie in movies:
        name = movie['short_name']
        print name
        binary_index,binary_news,digit_index_list,digit_news_list = split('index/'+name+'.png')
        idxs[name] = (img2index(binary_index,digit_index_list,INDEX_GAP),img2index(binary_news,digit_news_list,NEWS_GAP))
        
    with open('idx.pickle','w') as f:
        pickle.dump(idxs,f)

#binary_index,binary_news,digit_index_list,digit_news_list = split('index/我爱灰太狼2.png'.decode('utf-8'))
#idx = img2index(binary_index,digit_index_list,INDEX_GAP)
#idx = img2index(binary_news,digit_news_list,NEWS_GAP)
#plt.plot(idx)
#index2file()