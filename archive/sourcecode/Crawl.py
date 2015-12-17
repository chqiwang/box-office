import pickle
import urllib2
import urllib
from bs4 import BeautifulSoup
from selenium import webdriver
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import datetime

def crawlmovie(node):
    contents = {}
    m = node.select("div.bd.doulist-subject")[0]
    contents['name'] = m.select('div.title')[0].a.string.strip()
    contents['img_url'] = m.select('div.post')[0].img['src'].strip()
    contents['url'] = m.select('div.post')[0].a['href'].strip()
    contents['rating_score'] = m.select('div.rating')[0].select('span.rating_nums')[0].string.strip()
    contents['rating_people'] = m.select('div.rating')[0].select('span')[2].string.strip()[1:-4]
    abstrct = []
    for child in m.select('div.abstract')[0].children:
        if child.string == None:
            continue
        abstrct.append(child.string.strip())
    contents['abstrct'] = abstrct
    contents['tickts_date'] = node.select("div.ft > div > blockquote")[0].text.strip()[3:]
    return contents
    
def crawlsubmovies(index):
    movies = []
    response = urllib2.urlopen('http://www.douban.com/doulist/1295618/?start='+str(index)+'&sort=seq&sub_type=')
    html = response.read()
    soup = BeautifulSoup(html,"lxml")
    lsts = soup.select('div.doulist-item')
    for m in lsts:
        contents = crawlmovie(m.div)
        movies.append(contents)
        
    return movies
        
def crawlmovies(start=0,end=25):
    if start == 0:
        movies = []
    else:
        with open('movies.txt') as f:
            movies = pickle.load(f)
    try:
        for i in range(start,end+1):
            submovies = crawlsubmovies(i*25)
            movies += submovies
            print i,
    except Exception as e:
        print e
        print 'Error,restart'
    finally:
        with open('movies.txt','w') as f:
            pickle.dump(movies,f)

def crawlinfo(url):
    info = {}
    response = urllib2.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html,"lxml")
    interest =  soup.select('div.subject-others-interests-ft > a')
    info['watched'] = interest[0].string.strip()[:-3]
    info['will'] = interest[1].string.strip()[:-3]
    info['tags'] = [tag.string.strip() for tag in soup.select('div.tags-body > a')]
    info['infos'] = soup.select('div#info')[0].text.strip()
    info['scroes'] = [r.string.strip() for r in soup.select('span.rating_per')]
    info['greater'] = [g.string.strip() for g in soup.select('div.rating_betterthan > a')]
    info['story'] = soup.select('span[property="v:summary"]')[0].text.strip()
    if len(soup.select('span.all.hidden')) != 0:
        info['story'] = soup.select('span.all.hidden')[0].text.strip()    
    info['recomand'] = [rec.dd.a.string.strip() for rec in soup.select('div.recommendations-bd > dl')]
    return info
        
def crawlinfos(start = 0):
    with open('movies.txt') as f:
        movies = pickle.load(f)
    
    try:
        k = start
        for movie in movies[start:]:
            movie.update(crawlinfo(movie['url']))
            print k,movie['name']
            k += 1
    except Exception as e:
        print e
        print 'Error,restart at ' + str(k)
    finally:
        with open('movies.txt','w') as f:
            pickle.dump(movies,f)

def parsemovie(movie):
    parsedmovie = {}
    parsedmovie['related_movie'] = [name.split(' ')[0] for name in movie['recomand']]
    parsedmovie['rating_people_num'] = int(movie['rating_people'])
    parsedmovie['name'] = movie['name'].split(' ')[0]
    parsedmovie['tags'] = movie['tags']
    parsedmovie['per_rating_5to1'] = [float(s[:-1]) for s in movie['scroes']]
    parsedmovie['story'] = movie['story'].replace('\n','').replace('\t','').replace('  ','')
    parsedmovie['will_watch'] = int(movie['will'])
    parsedmovie['watched'] = int(movie['watched'])
    parsedmovie['rating_score'] = float(movie['rating_score'])
    infos = [info.strip().split("：".decode('utf-8')) for info in movie['tickts_date'].split('|')]
    infos = [g[1].strip() for g in infos if len(g) > 1]
    money = infos[0].split('（'.decode('utf-8'))[0].strip()[:-2]
    if '亿元'.decode('utf-8') in infos[0]:
        parsedmovie['total_money'] = int(float(money)*1000)
    else:
        parsedmovie['total_money'] = int(money)
    parsedmovie['date'] = infos[1]
    parsedmovie['product_type'] = infos[2].split('（'.decode('utf-8'))[0]
    infos = [info.split(": ".decode('utf-8'))[1].strip() for info in movie['abstrct']]
    k = 0
    if '导演'.decode('utf-8') in movie['abstrct'][k]:
        parsedmovie['director'] = infos[k]
        k += 1
    else:
        parsedmovie['director'] = 'None'
    if '主演'.decode('utf-8') in movie['abstrct'][k]:
        parsedmovie['actors'] = [act.strip() for act in infos[k].split('/')]
        k += 1
    else:
        parsedmovie['actors'] = ['None']
    parsedmovie['types'] = [t.strip() for t in infos[k].split('/')]
    parsedmovie['country'] = infos[k+1]
    parsedmovie['img_name'] = movie['img_url'].split('/')[-1].strip()
    parsedmovie['img_url'] = movie['img_url']
    greater = [g.split(' ') for g in movie['greater']]
    parsedmovie['better_than'] = {g[1].strip():int(g[0].strip()[:-1]) for g in greater}
    infos = [info.strip().split(": ".decode('utf-8')) for info in movie['infos'].split('\n')]
    dic = {info[0].strip():info[1].strip() for info in infos if len(info) > 1}
    if '语言'.decode('utf-8') in dic.keys():
        parsedmovie['language'] = dic['语言'.decode('utf-8')]
    else:
        parsedmovie['language'] = 'None'
    if '片长'.decode('utf-8') in dic.keys() and '分钟'.decode('utf-8') in dic['片长'.decode('utf-8')]:
        parsedmovie['length'] = int(dic['片长'.decode('utf-8')].split('/')[0].split('(')[0].strip()[:-2])
    else:
        parsedmovie['length'] = 0
    if '编剧'.decode('utf-8') in dic.keys():
        parsedmovie['authors'] = [author for author in dic['编剧'.decode('utf-8')].split('/')]
    else:
        parsedmovie['authors'] = ['None']
    parsedmovie['language'] = [l.strip() for l in parsedmovie['language'].split('/')]
    parsedmovie['country'] = [c.strip() for c in parsedmovie['country'].split('/')]
    
    if '&' not in parsedmovie['date']:
        parsedmovie['date'] = parsedmovie['date'].split('（'.decode('utf-8'))[0].strip()
    if '-' in parsedmovie['date']:
        d_e = datetime.datetime.strptime(parsedmovie['date'],'%Y-%m-%d').date()
    else:
        if '&' not in parsedmovie['date']:
            [year,rest] = parsedmovie['date'].split('年'.decode('utf-8'))
        else:
            [year,rest] = parsedmovie['date'].split('&')[1].split('（'.decode('utf-8'))[0].strip().split('年'.decode('utf-8'))
        [month,rest] = rest.split('月'.decode('utf-8'))
        day = rest.split('日'.decode('utf-8'))[0]
        d_e = datetime.date(int(year),int(month),int(day))
    parsedmovie['date'] = str(d_e)
    parsedmovie['short_name'] = parsedmovie['name'].split('：'.decode('utf-8'))[0].strip()
    return parsedmovie
    
def parsemovies():
    with open('movies.txt') as f:
        movies = pickle.load(f)
    
    parsedmovies = [parsemovie(m) for m in movies]
    
    with open('movies.txt','w') as f:
            pickle.dump(parsedmovies,f)

def crawlimages():
    with open('movies.txt') as f:
        movies = pickle.load(f)
    for movie in movies:
        print movie['name'],'downloading.....'
        urllib.urlretrieve(movie['img_url'],'imgs/'+movie['img_name'])

def crawl_search_index(driver,name,date):
    driver.get('http://index.baidu.com/?tpl=trend&type=0&area=0&time='+date[0]+'%7C'+date[1]+'&word=%B5%C1%C3%CE%BF%D5%BC%E4')
    driver.implicitly_wait(15)
    ele = driver.find_element_by_class_name('comWord')
    ele.clear()
    ele.send_keys(name)
    driver.find_element_by_class_name('compOK').click()
    ActionChains(driver).key_down(Keys.PAGE_DOWN).perform()
    time.sleep(5)
    exit()
    #driver.save_screenshot('index/'+name+'.png')
    
def crawl_search_indexes(movies,driver):
    global start
    min_d = datetime.date(2011,1,1)    
    for movie in movies[start:]:
        d_e = datetime.datetime.strptime(movie['date'],'%Y-%m-%d').date
        d_s = d_e - datetime.timedelta(30)
        if d_e < min_d:
            d_e = min_d
        if d_s < min_d:
            d_s = min_d
        crawl_search_index(driver,movie['short_name'],[str(d_s).replace('-',''),str(d_e).replace('-','')])
        print start,movie['name']
        start += 1

def crawl_index_with_except():
    with open('movies.txt') as f:
        movies = pickle.load(f)
    
    driver = webdriver.Chrome()
    driver.get('http://index.baidu.com/?tpl=trend&type=0&area=0&time=20150801%7C20150831&word=%B5%C1%C3%CE%BF%D5%BC%E4')
    driver.implicitly_wait(15)
    driver.find_element_by_id('TANGRAM_13__userName').send_keys('294254367@qq.com')
    driver.find_element_by_id('TANGRAM_13__password').send_keys('whkwdy8101971')
    driver.find_element_by_id('TANGRAM_13__submit').click()
    time.sleep(15)
    
    global start
    start = 0
    while start < len(movies):
        try:
            crawl_search_indexes(movies,driver)
        except Exception as e:
            print e
            print 'Error:start from ',start
            time.sleep(3600)
    
    driver.close()

def fixdates():
    with open('movies.txt') as f:
        movies = pickle.load(f)
    for movie in movies:
        if '&' not in movie['date']:
            movie['date'] = movie['date'].split('（'.decode('utf-8'))[0].strip()
        if '-' in movie['date']:
            d_e = datetime.datetime.strptime(movie['date'],'%Y-%m-%d').date()
        else:
            if '&' not in movie['date']:
                [year,rest] = movie['date'].split('年'.decode('utf-8'))
            else:
                [year,rest] = movie['date'].split('&')[1].split('（'.decode('utf-8'))[0].strip().split('年'.decode('utf-8'))
            [month,rest] = rest.split('月'.decode('utf-8'))
            day = rest.split('日'.decode('utf-8'))[0]
            d_e = datetime.date(int(year),int(month),int(day))
        movie['date'] = str(d_e)
        movie['short_name'] = movie['name'].split('：'.decode('utf-8'))[0].strip()
    with open('movies.txt','w') as f:
        pickle.dump(movies,f)

def fixlanguage():
    with open('movies.txt') as f:
        movies = pickle.load(f)
    for movie in movies:
        movie['language'] = [l.strip() for l in movie['language'].split('/')]
        movie['country'] = [c.strip() for c in movie['country'].split('/')]
    with open('movies.txt','w') as f:
        pickle.dump(movies,f)

def pickle2csv():
    with open('movies.txt') as f:
        movies = pickle.load(f)
    with open('movies.csv','w') as f:
        keys = movies[0].keys()
        f.write(str(keys)+'\n')
        for movie in movies:
            s = ''
            for key in keys:
                if key == 'story' or key == 'short_name' or key == 'director' or key == 'product_type' or key == 'name':
                    s += movie[key].encode('utf-8') + '$'
                elif key == 'per_rating_5to1':
                    sr = ''
                    for r in movie[key]:
                        sr += str(r) + ','
                    s += sr + '$'
                elif key == 'actors' or key == 'tags' or key == 'authors' or key == 'types' or key == 'language' or key == 'country' or key == 'related_movie':
                    sa = ''
                    for a in movie[key]:
                        sa += a.encode('utf-8') + ','
                    s += sa + '$'
                elif key == 'better_than':
                    sd = ''
                    for k in movie[key].keys():
                        sd += k.encode('utf-8') + ':' + str(movie[key][k]) + ','
                    s += sd + '$'
                else:
                    s += str(movie[key]) + '$'
            f.write(s + '\n')

#crawlmovies()
#crawlinfos()
#parsemovies()
#fixdates()
#fixlanguage()
#crawlimages()
#crawl_index_with_except()
#pickle2csv()