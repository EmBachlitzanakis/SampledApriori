import pandas as pd
import matplotlib.pyplot as plt
import random
import csv
import re
import time
import draw_rules_graph as graph
from scipy.stats import linregress
import numpy as np
import traceback
import datetime
import keyboard
import sys

#1
def CreateMovieBaskets(filename):
    df = pd.read_csv(filename, delimiter=',')
    # User list comprehension to create a list of lists from Dataframe rows
    list_of_rows = [list(row) for row in df.values]
    userBaskets = {}
    for elem in list_of_rows:
        if elem[0] not in userBaskets:
            userBaskets[elem[0]] = []
        userBaskets[elem[0]].append(elem[1:])
    #print(userBaskets[1])
    return userBaskets

#2
def readMovies():
    movies_df = pd.DataFrame(pd.read_csv('movies.csv'),columns=['movieId','title','genres'])
    return movies_df

#3.a
def pos(i,j,n):
    return int((i-1)*(n-i/2)+j-i)

def TriangularMatrixOfPairsCounters(moviesdf):
    user_baskets = CreateMovieBaskets('ratings_100users.csv')
    #moviesdf = readMovies()
    #moviesdf = moviesdf['movieId']
    movies = {}
    c = 1
    for id in moviesdf:
        movies[id] = c
        c +=1
    l = len(movies)           #plhtos tainiwn
    pairs = [0 for i in range((l*(l-1)//2)+1)]
    for i in range(1, 10):
        for j in range(len(user_baskets[i])):
            for k in range(j+1, len(user_baskets[i])):
               p = pos(movies[user_baskets[i][j][0]], movies[user_baskets[i][k][0]], l)
               pairs[p] += 1
    #print(pairs[:100])
    return pairs

def movie_baskets(user_baskets):
    moviebaskets = []
    for user in user_baskets:
        moviebaskets.append([])
        for basket in user_baskets[user]:  #basket[0] : movieId
            moviebaskets[-1].append(basket[0])
    return moviebaskets

#3.b
def HashedCountersOfPairs(moviebaskets, tuples): #tuples = [[tuple_1], [tuple_2], ...] (e.g. tuples)
                                                       #or
                                                       #tuples = [movie_1, movie_2, ...]     (e.g. single set)
    Lx = {}
    if(type(tuples[0]) is int):
        for basket in moviebaskets:
            for i in range(len(basket)):
                try:
                    Lx[basket[i]] += 1
                except:
                    Lx[basket[i]] = 1
    else:
        for tpl in tuples:
            ln = len(tpl)
            Lx[tuple(tpl)] = 0
            for basket in moviebaskets:
                x = len(set(tpl).intersection(basket))
                if(x == ln):
                    Lx[tuple(tpl)] += 1
    #print("--- %s seconds ---" % (time.time() - start_time))
    return Lx 

def myApriori(movies, itemBaskets, min_frequency, max_length):
    print("apriori")
    #start_time = time.time()
    tuples = []         # array of [[2-tuples], [3-tuples], ..., [k-tuples]]
    #find single-tuples
    L1all = HashedCountersOfPairs(itemBaskets, movies)
    L1 = []
    for movie in movies:
        try:
            frequency = L1all[movie] / len(itemBaskets)
            if(frequency >= min_frequency):
                L1.append(movie)
        except:
            pass
    #find 2-tuples
    L2all = []
    for i in range(len(L1)-1):
        for j in range(i+1, len(L1)):
            L2all.append([L1[i], L1[j]])
    try:
        L2dict = HashedCountersOfPairs(itemBaskets, L2all)
    except:
        print("L1: " + str(L1))
        return tuples    
    L2 = []
    for movie in L2all:
        frequency = L2dict[tuple(movie)] / len(itemBaskets)
        if(frequency >= min_frequency):
            L2.append(tuple(movie))
    print("yay")
    print("L1: " + str(L1))
    print("L2: " + str(L2))
    Li = L2
    Lx = []
    tuples.append(L2)
    Flag = True
    if(len(L2) < 2):
        Flag = False
    #find k-tuples, k >= 3
    k = 2
    while(Flag):
        for i in range(len(Li)):
            for j in range(len(L1)):
                if(L1[j] not in Li[i]):
                    l = []
                    l = list(Li[i])
                    l.append(L1[j])
                    Lx.append(l)
        Lxdict = HashedCountersOfPairs(itemBaskets, Lx)
        Li = []
        for movie in Lxdict:
            frequency = Lxdict[movie] / len(itemBaskets)
            if(frequency >= min_frequency):
                Li.append(movie)
        if(len(Li) != 0):
            tuples.append(Li)
        if((len(Li) < 2) or (k == max_length)):
            Flag = False
        Lx = []
        k+=1
    with open('tuples.txt', 'w') as fp:
        fp.write(str(tuples))
    fp.close()
   # print("--- %s seconds ---" % (time.time() - start_time))
    return tuples

def callApriori(movies, sample, min_frequency, max_length):
    samplemoviebaskets = []
    for key in sample:
        s = sorted(sample[key])
        samplemoviebaskets.append(s)
    print(samplemoviebaskets)
    return myApriori(movies, samplemoviebaskets, min_frequency, max_length)

#5
def sampledApriori(movies, stream, SampledScores, min_frequency, max_length):
    sample = {} #{user1 : [basket], ...}
    setofusers = []
    while(True):#moviebaskets and len(sample) < SampledScores):
        if((not stream) or (len(sample) == SampledScores)):
            break
        try:            #user in set of users
            sample[stream[0][0]].append(stream[0][1])
        except:
            setofusers.append(stream[0][0])
            sample[stream[0][0]] = [stream[0][1]]
        stream.pop(0)
    print("Type \'y\'/\'Y\' to end process.")
    c = 1;
    i = SampledScores
    timestocallapriori = (len(stream))//4
    while(stream):
        if keyboard.is_pressed('y') or keyboard.is_pressed('Y'):
            print("\'Y\' pressed. Process terminated...Calling Apriori...")
            break
        try:
            k = sample[stream[0][0]]  # user in set of users
            sample[stream[0][0]].append(stream[0][1])
        except:
            ran = random.randrange(i+1);
            if(ran < SampledScores):
                usertobedropped = setofusers[ran]
                setofusers[ran] = stream[0][0]
                del sample[usertobedropped]
                sample[stream[0][0]] = [stream[0][1]]    
            i+=1
        c+=1
        stream.pop(0)
    tuples = callApriori(movies, sample, min_frequency, max_length)
    return tuples

#6
def add_rules(tpl, lift, confidence, frequency_v_u, frequency_u, u, v, ruleid):
    interest = confidence - frequency_u
    l = [u]
    try:
        V = "["
        for i in range(len(v)):
            if(i == (len(v)-1)):
                V+= str(v[i]) + "]"
            else:
                V+= str(v[i]) + ","
    except:
        V = "[" + str(v) + "]"
    rule = V + " --> [" + str(u) + "]" 
    #infos.append(rule)
    infos = [list(tpl), v, l, rule, frequency_v_u, confidence, lift, interest, ruleid]
    return infos
    
def AssociationRulesCreation(moviebaskets, pos_sets, MinConfidence, MinLift, MaxLift): #pos_sets = {tuple_of_movies1: support1, tuple_of_movies2: support2, ...}
    minlift = True
    maxlift = True
    ruleid = 1
    if(MinLift == -1):
        minlift = False
    if(MaxLift == -1):
        maxlift = False
    rules_dict = {}
    for tpl in pos_sets: #v -> u
        v = list(tpl).copy()
        u = v.pop()
        l = []
        l.append(tuple(v))
        lx = HashedCountersOfPairs(moviebaskets, l)
        sup_v = lx[tuple(v)]
        sup_v_u = HashedCountersOfPairs(moviebaskets, [tpl])
        confidence = sup_v_u[tpl]/sup_v
        if(confidence >= MinConfidence):
            frequency_v_u = sup_v_u[tpl]/len(moviebaskets)
            frequency_v = sup_v/len(moviebaskets)
            l = []
            l.append(u)
            l = tuple(l)
            l2 = []
            l2.append(l)
            lx = HashedCountersOfPairs(moviebaskets, l2)
            sup_u = lx[l]
            frequency_u = sup_u/len(moviebaskets)
            lift = frequency_v_u/(frequency_v*frequency_u)
            if(minlift == True and maxlift == True):
                if(lift >= MinLift and lift <= MaxLift):
                    #infos = [hypothesis (v), conclusion (u), rule (v->u), frequency, confidence, lift, interest, ruleid]
                    infos = add_rules(tpl, lift, confidence, frequency_v_u, frequency_u, u, v, ruleid)
                    ruleid+=1
                    rules_dict[tpl] = infos
            elif(minlift == False and maxlift == True):
                if(lift <= MaxLift):
                    infos = add_rules(tpl, lift, confidence, frequency_v_u, frequency_u, u, v, ruleid)
                    ruleid+=1
                    rules_dict[tpl] = infos
            elif(minlift == True and maxlift == False):
                if(lift >= MinLift):
                    infos = add_rules(tpl, lift, confidence, frequency_v_u, frequency_u, u, v, ruleid)
                    ruleid+=1
                    rules_dict[tpl] = infos
            else:
                infos = add_rules(tpl, lift, confidence, frequency_v_u, frequency_u, u, v, ruleid)
                ruleid+=1
                rules_dict[tpl] = infos
    rules_df = pd.DataFrame.from_dict(rules_dict, orient='index', columns=["itemset", "hypothesis", "conclusion", "rule", "frequency", "confidence", "lift", "interest", "rule ID"])
    rules_df = rules_df.reset_index()
    rules_df = rules_df.drop(columns=['index'])
    return rules_df

#7
def format():
    inp = input("Provide your option: ")
    l = re.split(' ', inp)
    inp = "".join(l)
    l = re.split(',', inp)
    if(l[0] == 'a' and len(l) == 1):
        res = [1, l]
        return res
    elif(l[0] == 'b' and (l[1] == 'i' or l[1] == 'h' or l[1] == 'c')  and len(l) > 2):
        movies = len(l)-2
        m = ""
        for i in range(movies):
            if(not l[i+2].isnumeric()):
                print("Please enter a valid choice")
                format()
            if(i == movies-1):
                m += l[i+2]
            else:
                m += l[i+2]+ ", "
        print("Looking for rules containing movies {" + m + "}")
        return [2, l]
    elif(l[0] == 'c' and len(l) == 1):
        return [3, l]
    elif(l[0] == 'h' and (l[1] == 'c' or l[1] == 'l') and len(l) == 2):
        res = [4, l]
        return res
    elif(l[0] == 'm' and l[1].isnumeric() and len(l) == 2):
        res = [5, l]
        return res
    elif(l[0] == 'r' and l[1].isnumeric() and len(l) == 2):
        res = [6, l]
        return res
    elif(l[0] == 's' and (l[1] == 'c' or l[1] == 'l') and len(l) == 2):
        res = [7, l]
        return res
    elif(l[0] == 'v' and (l[1] == 'c' or l[1] == 'r' or l[1] == 's') and l[2].isnumeric() and len(l) == 3):
        res = [8, l]
        return res
    elif(l[0] == 'e' and len(l) == 1):
        res = [9, l]
        return res
    print("Please enter a valid choice")
    return format()

def presentResults(rules_df):
    f = open("menu.txt", "r")
    print(f.read())
    while(True):
        res = format()
        if(res[0] == 1):
            rules = rules_df['rule'].tolist()
            b = len(str(rules_df['rule ID'][len(rules_df)-1]))
            for i in range(len(rules_df)):
                spc = b - len(str(rules_df['rule ID'][i]))
                print("Rule ID " + str(rules_df['rule ID'][i]) + (" "*spc) + " :    " + rules_df['rule'][i])
        elif(res[0] == 2):
            movies = []
            for i in range(2, len(res[1])):
                movies.append(int(res[1][i]))
            if(res[1][1] == 'i'):
                itemsets = rules_df['itemset'].tolist()
                ruleIds = rules_df['rule ID'].tolist()
                i = 0
                ids = []
                for itemset in itemsets:
                    for movie in movies:
                        if(movie in itemset):
                            ids.append(ruleIds[i])
                            break
                    i+=1
                print("...........................................................................")
                if(not ids):
                    print("No itemsets found")
                else:
                    b = len(str(ids[-1]))
                    rules = rules_df[rules_df['rule ID'].isin(ids)]['rule'].tolist()
                    i = 0
                    for rule in rules:
                        spc = b - len(str(ids[i]))
                        print("Rule ID " + str(ids[i]) + (" "*spc) + " :    " + rule)
                        i+=1  
                print("...........................................................................")
            elif(res[1][1] == 'h'):
                hypotheses = rules_df['hypothesis'].tolist()
                ruleIds = rules_df['rule ID'].tolist()
                i = 0
                ids = []
                for hypothesis in hypotheses:
                    for movie in movies:
                        if(movie in hypothesis):
                            ids.append(ruleIds[i])
                            break
                    i+=1
                
                print("...........................................................................")
                if(not ids):
                    print("No itemsets found")
                else:
                    b = len(str(ids[-1]))
                    rules = rules_df[rules_df['rule ID'].isin(ids)]['rule'].tolist()
                    i = 0
                    for rule in rules:
                        spc = b - len(str(ids[i]))
                        print("Rule ID " + str(ids[i]) + (" "*spc) + " :    " + rule)
                        i+=1
                print("...........................................................................")
            else:
                conclusions = rules_df['conclusion'].tolist()
                ruleIds = rules_df['rule ID'].tolist()
                i = 0
                ids = []
                for conclusion in conclusions:
                    for movie in movies:
                        if(movie in conclusion):
                            ids.append(ruleIds[i])
                            break
                    i+=1
                print("...........................................................................")
                if(not ids):
                    print("No itemsets found")
                else:
                    b = len(str(ids[-1]))
                    rules = rules_df[rules_df['rule ID'].isin(ids)]['rule'].tolist()
                    i = 0
                    for rule in rules:
                        spc = b - len(str(ids[i]))
                        print("Rule ID " + str(ids[i]) + (" "*spc) + " :    " + rule)
                        i+=1
                print("...........................................................................")
        elif(res[0] == 3):
            fit = np.polyfit(rules_df['lift'], rules_df['confidence'], 1)
            fit_fn = np.poly1d(fit)
            plt.plot(rules_df['lift'], rules_df['confidence'], 'yo', rules_df['lift'],
            fit_fn(rules_df['lift']))
            plt.xlabel('Lift')
            plt.ylabel('Confidence')
            plt.title('CONFIDENCE vs LIFT')
            plt.tight_layout()
            plt.show() 

        elif(res[0] == 4):
            if(res[1][1] == 'c'):
                hist = rules_df.hist(column='confidence')
                plt.show()
            else:
                hist = rules_df.hist(column='lift')
                plt.show()
        elif(res[0] == 5):
            moviesdf = readMovies()
            resdf = moviesdf[moviesdf['movieId'] == int(res[1][1])]
            print("...........................................................................")
            print(resdf.iloc[0])
            print("...........................................................................")
        elif(res[0] == 6):
            resdf = rules_df[rules_df['rule ID'] == int(res[1][1])]
            resdf = resdf.reset_index()
            print("...........................................................................")
            print("Rule ID " + str(resdf['rule ID'][0])+ " :    " + str(resdf['rule'][0]))
            print("...........................................................................")
        elif(res[0] == 7):
            print("...........................................................................")
            if(res[1][1] == 'c'):
                resdf = rules_df.sort_values(by=['confidence'])
                resdf = resdf.reset_index()
                for i in range(len(resdf)):
                    print("Rule:   " + str(resdf['rule'][i]) + ", confidence:" + str(resdf['confidence'][i]))
            else:
                resdf = rules_df.sort_values(by=['lift'])
                resdf = resdf.reset_index()
                for i in range(len(resdf)):
                    print("Rule:   " + str(resdf['rule'][i]) + ", lift:" + str(resdf['lift'][i]))
            print("...........................................................................")
        elif(res[0] == 8):
            resdf = rules_df[rules_df['rule ID'] <= int(res[1][2])]
            if(res[1][1] == 'c'):
                graph.draw_graph(resdf, 'c')
            elif(res[1][1] == 'r'):
                graph.draw_graph(resdf, 'r')
            else:
                graph.draw_graph(resdf, 's')
        elif(res[0] == 9):
            print("Exiting...")
            break
    
    
    
        
    
######
#Main#
######

try:
    sys.argv[1]
except:
    print("No filename given. Format: \" python bigdata2.py filename\".")
moviesdf = readMovies()
movies = moviesdf['movieId']
movies = movies.values.tolist()

user_baskets = CreateMovieBaskets(sys.argv[1])
moviebaskets = movie_baskets(user_baskets)
tpls = myApriori(movies, moviebaskets, 0.3, 4)
print("done, tuples:")
tuples = []
for i in range(len(tpls)):
    for j in range(len(tpls[i])):
        tuples.append(tpls[i][j])

df = AssociationRulesCreation(moviebaskets, tuples, 0.1, -10, 10)

print("DataFrame: ")
print(df)
print("...........................................................................")
presentResults(df)


##################################
#Kwdikas peiramatikhs a3iologhshs#
##################################
"""
user_baskets_ap = CreateMovieBaskets('ratings.csv')
moviebaskets_ap = movie_baskets(user_baskets_ap)
tpls = myApriori(movies, moviebaskets_ap, 0.19, 2)
print("done, tuples:")
tuples_ap = []
for i in range(len(tpls)):
    for j in range(len(tpls[i])):
        tuples_ap.append(tpls[i][j])
print(tuples_ap)


df = pd.read_csv('ratings_shuffled.csv', delimiter=',')
stream =[] 
for index, rows in df.iterrows(): 
    my_list =[rows.userId, rows.movieId] 
    stream.append(my_list)  

tpls = sampledApriori(movies, stream, 100, 0.19, 2)
print("done, tuples:")
tuples_sam = []
for i in range(len(tpls)):
    for j in range(len(tpls[i])):
        tuples_sam.append(tpls[i][j])
print(tuples_sam)

#TRUE POSITIVES:
tp = len(set(tuples_sam).intersection(tuples_ap)) #osa yparxoyn kai sto deigma kai sto pragmatiko synolo (dhladh h tomh tous)
#FALSE POSITIVES:
fp = len(tuples_sam) - tp                         #osa yparxoyn sto deigma alla oxi sto pragmatiko synolo 
#FALSE NEGATIVES:
fn = len(tuples_ap) - tp                          #osa yparxoyn sto pragmatiko synolo alla den yparxoyn sto deigma

p = tp / (tp + fp)  #prepei na nai 1
r = tp / (tp + fn)

#########
#METRICS#
#########
print("---------------------------------")
print("TRUE POSITIVES : " + str(tp))
print("FALSE POSITIVES: " + str(fp))
print("FALSE NEGATIVES: " + str(fn))
print("---------------------------------")
print("PRECISION:       " + str(p))
print("RECALL   :       " + str(r))
try:
    f1 = 2*r*p / (r+p)
    print("F1       :       " + str(f1))
except:
    print("Can't calculate F1 because PRECESION and RECALL equals '0'.")
print("---------------------------------")
"""
