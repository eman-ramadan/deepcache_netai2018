"""
                           DeepCache
               
DeepCache is distributed under the following BSD 3-Clause License:

Copyright(c) 2019
                University of Minensota - Twin Cities
        Authors: Arvind Narayanan, Saurabh Verma, Eman Ramadan, Pariya Babaie, and Zhi-Li Zhang

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


@author: Pariya Babaie (babai008@umn.edu) & Eman Ramadan (eman@cs.umn.edu)


DESCRIPTION:
    The code is based on
    Tang, Wenting, et al. "MediSyn: A synthetic streaming media service workload generator."
    Proceedings of the 13th international workshop on Network and operating systems support
    for digital audio and video. ACM, 2003.
    https://dl.acm.org/citation.cfm?id=776327

    The variables are mostly based on the paper, but we did some changes to generate requests which fit our need as
    shown in the request analysis plots. In addition to object popularity, and request interarrival distribution, this
    dataset simulates real traffic by:
        1- having new objects introduced at different times
        2- objects have different types with variable lifespans
        3- requests for each object are generated based on an hourly request ratio


INPUT:
    The inputs are hard coded and initialized based on the paper.
    You can modify the variables in the {initialization} section @Line 50


FLOW:
    1- Load the diurnal ratios for hourly rates, and regenerate them if GENERATE_NEW_HOURLY_REQUEST_RATIO is True
    2- Generate the days of introduction for objects, number of objects generated per day, and the
       interarrival time between objects introduced each day.
    3- Generate object frequencies.
    4- Generate objects and their properties, lifespan, introduction-time, and end-time.
    5- Generate and export the requests.


OUTPUT:
    The output is generated in '../Datasets' directory with the name:
     'mediSynDataset_x{'hourly_request_function_degree( i.e. 2)'}_O{'NUM_OF_OBJECTS'}.csv'
    the requests file contains object_IDs to be requested and their request_time.
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import random
import math
import csv
import sys
import os


""""###########################  initialization  ####################################################################"""
NUM_OF_OBJECTS = 3500                             # number of objects to generate
lambdaFile = 'hourly_request_ratio.csv'         # file with hourly request ratio
lambdas = []                                    # stores the diurnal ratios for non-homogeneous Poisson
curTime = []                                    # for each object shows the last timeStamp that it was requested
objectPopularities = []                         # Contains object popularity
M = 178310                                      # max frequency for HPC dataset
traceType = 'HPC'
hourly_request_function_degree = 2              # the degree for the function that sets the objects per bin pattern, X^2
dayGaps = []                                    # interarrival between days
numOfObjectsIntroduced = []                     # number of objects generated in each day
interArrivals = []                              # generates the interarrival time between objects introduced in a day
lifeSpanType = []                               # for each object it holds the type of its lifeSpan
ObjectsLifeSpan = []                            # the length of lifeSpan value for each object
requestGenInfo = {}                             # for each object it holds the info about requests
startTimes = {}                                 # sorted objects based on their introduction time
introductionOrder = []                          # random order for introducing objects in a day
sortedOnIntoTime = [] 
requests = []                                   # generated requests
objectLengths = []
if sys.version_info[0] < 3:
    maxEndDay = -sys.maxint - 1
else:
    maxEndDay = -sys.maxsize - 1
WITH_INTRODUCTION = True                        # flag to allow objects to be introduced at any time
WITH_DAY_GAPS_INTRODUCTION = False              # If True, introduce gaps between the objects introduction days,
                                                # otherwise objects are introduced each day
GENERATE_NEW_HOURLY_REQUEST_RATIO = False       # If True, a new 'hourly_request_ratio.csv' is generated
MIN_REQ_PER_DAY_THRESHOLD = 1500                # min number of requests to be generated for each object in a day
MIN_OBJ_INTRODCUED_PER_DAY_THRESHOLD = 0.0035 * NUM_OF_OBJECTS  # min number of objects to be generated in a day
MAX_OBJ_INTRODCUED_PER_DAY_THRESHOLD = 0.0095 * NUM_OF_OBJECTS  # max number of objects to be generated in a day

# Creating output directory if it does not exist
OUTPUTDIR = '../Datasets'
if not os.path.isdir(OUTPUTDIR):
    os.mkdir(OUTPUTDIR)

# Checking the existence of hourly_request_ratio.csv file
if not os.path.isfile('hourly_request_ratio.csv'):
    GENERATE_NEW_HOURLY_REQUEST_RATIO = True

if GENERATE_NEW_HOURLY_REQUEST_RATIO:
    print('Generating hourly request ratio file ...')
    rands = np.random.randint(1, 100, 24)
    rands = rands/float(np.sum(rands))
    index = np.arange(1, 25)

    res = 'hourly_request_ratio.csv'
    f = open(res, 'w+')
    for i in range(len(index)):
        if i != len(index)-1:
            f.write(str(index[i]) + ',' + str(rands[i])+'\n')
        else:
            f.write(str(index[i]) + ',' + str(rands[i]))
    f.close()


def initialize():
    global curTime
    loadDiurnalRatios()
    print('Generating Objects for Dataset ...')
    generateObjectsIntroductionInfo(traceType)
    generatePopularities(traceType, int(NUM_OF_OBJECTS))
    generateObjects()

    print('Generating Requests for Dataset ...')
    curTime = [0] * NUM_OF_OBJECTS
    generateRequests()


"""################################ Load diurnal ratios #############################################################"""
def loadDiurnalRatios():
    with open(lambdaFile, "r+") as fi:
        for line in fi:
            tmpLambdas = float(line.rstrip('\n').rstrip('\r').split(',')[1])
            lambdas.append(tmpLambdas)
    fi.close()


"""###########################  Object Popularity  ##################################################################"""
K = {'HPC': 30, 'HCL': 7}


def generatePopularities(traceType, N):
    zipalpha = 0.8
    k = K[traceType]
    for i in range(1, N+1):
        Mk = ((M-1)/k)+1
        tmp = (((float(Mk)/(math.pow((float(i+k-1)/k), zipalpha)))-1)*k)+1
        objectPopularities.append(tmp)


"""########################  Object Type  ###########################################################################"""
def getObjectType():
    decision = random.uniform(0, 1)
    if decision <= 0.1:  # 10 % of objects are news
        return 'news'
    else:
        return 'regular'


"""##################### generating random variates #################################################################"""
def generatePoissonVariate(rand, lambda_poisson):
    """
    for diurnal access generation
    """
    return -1 * (math.log(1-rand))/lambda_poisson


def generateParetoVariate(rand, alpha):
    return math.pow(1/rand, 1/alpha)


def generateParetoScaledVariate(rand, alpha, beta):
    """ F(x) = 1 - (b/x)^a, x >= b """
    return beta / (math.pow((1 - rand), (1/alpha)))


def generateNormalVariate(mu, sigma):
    """
    RV generated using rejection method 
    """
    variateGenerated = False
    while not variateGenerated:
        u1 = random.uniform(0, 1)
        u2 = random.uniform(0, 1)
        x = -1*math.log(u1)
        if u2 > math.exp(-1*math.pow((x-1), 2)/2):
            continue
        else:
            u3 = random.uniform(0, 1)
            if u3 > 0.5:
                return mu+(sigma*x)
            else:
                return mu-(sigma*x)


def generateLogNormalVariate(mu, sigma):
    """
    RV generated using rejection method 
    """
    variateGenerated = False
    while not variateGenerated:
        u1 = random.uniform(0, 1)
        u2 = random.uniform(0, 1)
        x = -1*math.log(u1)
        if u2 > math.exp(-1*math.pow((x-1), 2)/2):
            continue
        else:
            return math.exp(mu+(sigma*x))


def generateExponentialVariate(rand, a):
    return -(1/a)*math.log(1-rand)


def generateRandVariate(dist, params, numOfVariates):
    variates = []
    
    if dist is 'pareto':
        alpha = params['alpha']
        for i in range(numOfVariates):
            rand = random.uniform(0, 1)
            variates.append(generateParetoVariate(rand, alpha))

    if dist is 'paretoScaled':
        alpha = params['alpha']
        beta = params['beta']
        for i in range(numOfVariates):
            rand = random.uniform(0, 1)
            variates.append(generateParetoScaledVariate(rand, alpha, beta))

    elif dist is 'normal':
        mu = params['mu']
        sigma = params['sigma']
        for i in range(numOfVariates):
            variates.append(generateNormalVariate(mu, sigma))
            
    elif dist is 'logNormal':
        mu = params['mu']
        sigma = params['sigma']
        for i in range(numOfVariates):
            variates.append(generateLogNormalVariate(mu, sigma))
            
    elif dist is 'exp':
        mu = params['mu']
        for i in range(numOfVariates):
            rand = random.uniform(0, 1)
            variates.append(generateExponentialVariate(rand, mu))
    elif dist is 'poisson':
        mu = params['mu']
        for i in range(numOfVariates):
            rand = random.uniform(0, 1)
            variates.append(generatePoissonVariate(rand, mu))
    return variates


"""####################  Object Introduction Info  ##################################################################"""
def generateObjectsIntroductionInfo(typeMode):
    """
    generates gaps between introduction days based on either pareto or exponential distribution
    """
    global NUM_OF_OBJECTS
    global numOfObjectsIntroduced

    tempNumOfObjectsIntroduced = []
    while sum(tempNumOfObjectsIntroduced) < NUM_OF_OBJECTS:
        if typeMode is 'HPC':
            if WITH_DAY_GAPS_INTRODUCTION:
                pareto_alpha_objectIntro_hpc = 1.0164
                object_intro_days_gap = generateRandVariate('pareto', {'alpha':pareto_alpha_objectIntro_hpc}, 1)[0]
                if object_intro_days_gap > 20:
                    object_intro_days_gap = 20
                dayGaps.append(object_intro_days_gap)
            else:
                dayGaps.append(1)
            
        else:
            exponential_mu_objectIntro_hpl = 4.2705
            object_intro_days_gap = generateRandVariate('exp', {'mu': exponential_mu_objectIntro_hpl}, 1)[0]
            dayGaps.append(object_intro_days_gap)
        
        # number of new objects generated in each introduction day Pareto dist
        pareto_alpha_numOfObjectsGeneration = 0.8
        pareto_beta_numOfObjectsGeneration = MIN_OBJ_INTRODCUED_PER_DAY_THRESHOLD
        numOfObjects_intro_in_day = generateRandVariate('paretoScaled', {'alpha': pareto_alpha_numOfObjectsGeneration,
                                                        'beta': pareto_beta_numOfObjectsGeneration}, 1)[0]
        if numOfObjects_intro_in_day > MAX_OBJ_INTRODCUED_PER_DAY_THRESHOLD:
            numOfObjects_intro_in_day = MAX_OBJ_INTRODCUED_PER_DAY_THRESHOLD
        tempNumOfObjectsIntroduced.append(numOfObjects_intro_in_day)

    # sort generated items
    tempNumOfObjectsIntroduced.sort()
    extra_days = 0
    if len(tempNumOfObjectsIntroduced) % 7 != 0:
        extra_days = len(tempNumOfObjectsIntroduced) % 7
        for i in range(extra_days):
            # generate random int to add these objects to other introduction days to generate full weeks of data
            added = False
            while not added:
                u = random.randint(extra_days+1, len(tempNumOfObjectsIntroduced) - 1)
                if tempNumOfObjectsIntroduced[i] + tempNumOfObjectsIntroduced[u] < MAX_OBJ_INTRODCUED_PER_DAY_THRESHOLD:
                    tempNumOfObjectsIntroduced[u] += tempNumOfObjectsIntroduced[i]
                    added = True

    # Exclude the extra days after being added to other days
    tempNumOfObjectsIntroduced = tempNumOfObjectsIntroduced[extra_days:]
    tempNumOfObjectsIntroduced.sort()

    # Fill in the days by dividing the sorted data as following
    # This induces that more objects are introduced on Friday then Saturday, and so on.
    # The least number of objects are introduced on Tuesday.
    # Fri 1, Sat 2, Sun 3, Thu 4, Wed 5, Mon 6, Tuesday 7
    weeks = int(len(tempNumOfObjectsIntroduced) / 7)
    FriIndex = weeks * 6
    SatIndex = weeks * 5
    SunIndex = weeks * 4
    MonIndex = weeks * 1
    TuesIndex = weeks * 0
    WedIndex = weeks * 2
    ThuIndex = weeks * 3

    for i in range(weeks):
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[MonIndex+i])
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[TuesIndex + i])
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[WedIndex + i])
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[ThuIndex + i])
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[FriIndex + i])
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[SatIndex + i])
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[SunIndex + i])

    # interarrivalTime for objects introduction in a day
    pareto_alpha_interArrival = 1.0073
    numOfDays = len(numOfObjectsIntroduced)
    for i in range(numOfDays):
        objectsCountInDay = int(np.round(numOfObjectsIntroduced)[i])
        if WITH_INTRODUCTION:
            interArrivals.append(generateRandVariate('pareto', {'alpha': pareto_alpha_interArrival}, objectsCountInDay))
        else:
            interArrivals.append([0]*objectsCountInDay)
    NUM_OF_OBJECTS = int(sum(np.round(numOfObjectsIntroduced)))


def generateObjectIntroductionOrder():
    return np.random.permutation(range(len(objectPopularities)))+1


"""#########################  Object lifespan  ######################################################################"""
def generateLifeSpans(numOfObjects, objMode):
    logNormal_mu_mean = 3.0935
    logNormal_mu_std = 0.9612
    logNormal_sigma_mean = 1.1417
    logNormal_sigma_std = 0.3067
    pareto_alpha_mean = 1.7023
    pareto_alpha_std = 0.2092
    lifeSpans = []
   
    logNormalMu = generateRandVariate('normal', {'mu': logNormal_mu_mean, 'sigma': logNormal_mu_std}, 1)[0]
    logNormalSigma = generateRandVariate('normal', {'mu': logNormal_sigma_mean, 'sigma': logNormal_sigma_std}, 1)[0]
    
    paretoAlpha = generateRandVariate('normal', {'mu': pareto_alpha_mean, 'sigma': pareto_alpha_std}, 1)[0]
    
    for i in range(numOfObjects):
        if objMode[i] is 'regular':
            tmpLifeSpan = generateRandVariate('logNormal', {'mu': logNormalMu, 'sigma': logNormalSigma}, 1)[0]
        elif objMode[i] is 'news':
            tmpLifeSpan = generateRandVariate('pareto', {'alpha': paretoAlpha}, 1)[0]
        if tmpLifeSpan > 80:
            tmpLifeSpan = random.randint(2, 80)
        lifeSpans.append((i+1, tmpLifeSpan))
    return lifeSpans


"""#########################  Object Generation  ####################################################################"""
def normalizePopularities():
    normalized = np.array(objectPopularities)/max(objectPopularities)
    return normalized


def getBinInterval(time):
    return (math.floor(time/float(3600)))/float(23)


def generateObjects():
    global ObjectsLifeSpan
    global introductionOrder
    global sortedOnIntoTime
    global maxEndDay
    normalizedPop = normalizePopularities()
    
    for i in range(len(normalizedPop)):
         lifeSpanType.append(getObjectType())
    # tuple (objID, LifeSpan), objID from 1 to N
    ObjectsLifeSpan = generateLifeSpans(len(objectPopularities), lifeSpanType)
    introductionOrder = generateObjectIntroductionOrder()   # objectIntroductionOrder from 1 to N
    for i in range(1, len(objectPopularities)+1):
        requestGenInfo[i] = {'startDay': 0, 'lifeSpan': 0, 'endDay': 0, 'arrivalTime': 0, 'type': '', 'freq': 0,
                             'unitPerDay': 0} # From 1 to N
        startTimes[i] = 0
        
    objCnt = 0  
    dayCnt = 0
    for i in range(len(numOfObjectsIntroduced)):
        dayTime = 0
        dayCnt = dayCnt+round(dayGaps[i])
        for j in range(int(np.round(numOfObjectsIntroduced)[i])):
            objIntroduced = introductionOrder[objCnt]
            dayTime = dayTime+interArrivals[i][j]
            requestGenInfo[objIntroduced]['startDay'] = dayCnt
            requestGenInfo[objIntroduced]['arrivalTime'] = dayTime 
            requestGenInfo[objIntroduced]['lifeSpan'] = ObjectsLifeSpan[objIntroduced-1][1]
            requestGenInfo[objIntroduced]['type'] = lifeSpanType[objIntroduced-1]
            requestGenInfo[objIntroduced]['freq'] = objectPopularities[objIntroduced-1]

            # Generating at least a minimum number of requests per day
            if requestGenInfo[objIntroduced]['freq'] / requestGenInfo[objIntroduced]['lifeSpan'] \
                    < MIN_REQ_PER_DAY_THRESHOLD:
                # generate a random number for which number to update
                decision = random.uniform(0, 1)
                if decision <= 0.5:
                    # update the object frequency
                    life_span = random.randint(10, 80)
                    requestGenInfo[objIntroduced]['freq'] = life_span * MIN_REQ_PER_DAY_THRESHOLD
                    requestGenInfo[objIntroduced]['lifeSpan'] = life_span
                else:
                    # update the object life-span
                    freq = random.randint(MIN_REQ_PER_DAY_THRESHOLD, 80*MIN_REQ_PER_DAY_THRESHOLD)
                    requestGenInfo[objIntroduced]['freq'] = freq
                    requestGenInfo[objIntroduced]['lifeSpan'] = freq / MIN_REQ_PER_DAY_THRESHOLD

            startTimes[objIntroduced] = dayCnt+getBinInterval(dayTime)

            requestGenInfo[objIntroduced]['endDay'] = requestGenInfo[objIntroduced]['lifeSpan'] + \
                                                      requestGenInfo[objIntroduced]['startDay']
            requestGenInfo[objIntroduced]['totalDens'] = math.pow(requestGenInfo[objIntroduced]['lifeSpan'],
                                                                  hourly_request_function_degree)

            objectLengths.append([objIntroduced, requestGenInfo[objIntroduced]['startDay'],
                                  requestGenInfo[objIntroduced]['lifeSpan'], requestGenInfo[objIntroduced]['endDay'],
                                  requestGenInfo[objIntroduced]['freq']])

            if requestGenInfo[objIntroduced]['endDay'] > maxEndDay:
                maxEndDay = requestGenInfo[objIntroduced]['endDay']
            objCnt = objCnt+1
    
    sortedOnIntoTime = sorted(startTimes, key=startTimes.get)

    
def generateDiurnalAccess(obj, diurnalRatio, dayCnt):
    global requests
    
    lifeTimeLeft = requestGenInfo[obj]['lifeSpan']

    if lifeTimeLeft > 1:
        lastDay = requestGenInfo[obj]['endDay']
        objCount = abs(requestGenInfo[obj]['freq']*(((math.pow(dayCnt-lastDay, hourly_request_function_degree)
                       - math.pow(lastDay-dayCnt+1, hourly_request_function_degree)))/requestGenInfo[obj]['totalDens']))
        requestGenInfo[obj]['lifeSpan'] = requestGenInfo[obj]['lifeSpan']-1
        for i in range(len(diurnalRatio)):
            tmpCount = int(np.round(objCount*diurnalRatio[i]))
            if tmpCount != 0:
                tmpLambda = (tmpCount/float(3600))
                reqInterArrivals = generateRandVariate('exp', {'mu': tmpLambda}, tmpCount)
                for tmpInter in reqInterArrivals:
                    requests.append((obj, (curTime[obj-1]+tmpInter)))
                    curTime[obj-1] = curTime[obj-1]+tmpInter
            
    else:
        lastDay = requestGenInfo[obj]['endDay']
        objCount = abs(requestGenInfo[obj]['freq']*(((math.pow(lastDay-dayCnt, hourly_request_function_degree)
                       - math.pow(lastDay-(dayCnt+requestGenInfo[obj]['lifeSpan']), hourly_request_function_degree))) /
                                                    requestGenInfo[obj]['totalDens']))
        spanToGenerate = int(math.floor(requestGenInfo[obj]['lifeSpan']*10))
        requestGenInfo[obj]['lifeSpan'] = 0
        
        for i in range(spanToGenerate):
            tmpCount = int(np.round(objCount*diurnalRatio[i]))
            if tmpCount != 0:
                tmpLambda = (tmpCount/float(3600))
            
                reqInterArrivals = generateRandVariate('exp', {'mu': tmpLambda}, tmpCount)
                for tmpInter in reqInterArrivals:
                    requests.append((obj, (curTime[obj-1]+tmpInter)))
                    curTime[obj-1] = curTime[obj-1]+tmpInter


"""#########################  Requests Generation  ##################################################################"""
def generateRequests():
    global requests
    global curTime

    OUTPUTFILENAME = '{0}/mediSynDataset_x{1}_O{2}.csv'.format(OUTPUTDIR, hourly_request_function_degree, NUM_OF_OBJECTS)
    if not os.path.isfile(OUTPUTFILENAME):
        fi = open(OUTPUTFILENAME, 'w')
        fi.write('object_ID,request_time\n')
        fi.close()

    dayCount = requestGenInfo[sortedOnIntoTime[0]]['startDay']
    reqGendf = pd.DataFrame.from_dict(requestGenInfo, orient='index')
    reqGendf['objID'] = reqGendf.index

    while dayCount <= maxEndDay:
        objList = list(reqGendf[(reqGendf['startDay'] <= dayCount) & (reqGendf['endDay'] >= dayCount)]['objID'])
        for obj in objList:
            if curTime[obj-1] == 0:
                curTime[obj-1] = (dayCount*86400) + requestGenInfo[obj]['arrivalTime']

            generateDiurnalAccess(obj, lambdas, dayCount)
                
        dayCount = dayCount + 1
        if dayCount % 20 == 0:
            requests = sorted(requests, key=lambda x: x[1])
            saveRequestsToFile(OUTPUTFILENAME)
            requests = []
            print('{} Days Processed of {} Total Days'.format(dayCount, int(maxEndDay)))
    print('MediSyn Dataset Saved to Output file: {}'.format(OUTPUTFILENAME))


def saveRequestsToFile(OUTPUTFILENAME):
    with open(OUTPUTFILENAME, 'a') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(requests)


"""##################################################################################################################"""


def main():
    initialize()


if __name__ == "__main__": main()
