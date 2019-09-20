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


@author: Eman Ramadan (eman@cs.umn.edu)


DESCRIPTION:
    This code analyzes the generated requests and extracts the properties of each object such as: when it was first
    introduced, its frequency, its lifespan. It also extracts the number of active unique objects requested in each
    hour.


PREPROCESSING_SCRIPTS:
    Need to run any of these scripts before running the requestAnalysis.py script
    1. generateSyntheticDataset.py: generates a synthetic dataset
    2. generateMediSynDataset.py: generates a synthetic dataset according to MediSyn paper


INPUT:
    The input directory is '../Datasets':
    1- REQUESTFILENAME: The request file to be analyzed @Line 41
    2- FORCE_GENERATE_BINS: a flag to force the regeneration of the bin file, by default it is False
    3- FORCE_GENERATE_PROPERTIES: a flag to force the regeneration of the object properties, by default it is False


OUTPUT:
    The output files are generated in '../Datasets' directory:
    1- {RequestFile}_bins.csv: which indicates the number of unique objects in each hour.
                            Format: {binID, uniqueObjNum, binMinRequestTime, binMaxRequestTime}
    2- {RequestFile}_properties.csv: which includes the properties of each object.
                            Format: {object_ID, frequency, lifeSpan, minRequestTime, maxRequestTime, start_day, end_day}
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import os

FORCE_GENERATE_BINS = False
FORCE_GENERATE_PROPERTIES = False

REQDIR = '../Datasets/'
REQUESTFILENAME = 'mediSynDataset_x2_O3488.csv'  #'syntheticDataset_O50.csv'

REQUESTPATH = REQDIR + REQUESTFILENAME
BINFILENAME = REQDIR + REQUESTFILENAME[:-4] + '_bins.csv'
PROPERTIES_FILENAME = REQDIR + REQUESTFILENAME[:-4] + '_properties.csv'
BIN_SECONDS_WIDTH = 3600

# Load Requests File
print('Loading Request File ...')
reqdf = pd.read_csv(REQUESTPATH, sep=',')
print('Sorting Request File by time ...')
reqdf.sort_values(by=['request_time'], inplace=True)
print('Request File Sorted')

# get all 1-hour intervals/bins
if not os.path.isfile(BINFILENAME) or FORCE_GENERATE_BINS:
    bins = np.arange(np.ceil(reqdf.request_time.min()), np.ceil(reqdf.request_time.max()), BIN_SECONDS_WIDTH)
    print('Starting binning process ...')
    reqdf['binID'] = pd.cut(reqdf['request_time'], bins, labels=np.arange(0, len(bins)-1))

    grp = reqdf.groupby(['binID']).agg({'object_ID': {'uniqueObjNum': lambda x: x.nunique()},
                                        'request_time': ['min', 'max']})
    grp.reset_index(level=0, inplace=True)
    # clean up columns
    cols = list()
    for k in grp.columns:
        if k[1] == '':
            cols.append(k[0])
        else:
            cols.append(k[1])
    grp.columns = cols

    filtered = grp.dropna()
    filtered["uniqueObjNum"] = filtered["uniqueObjNum"].apply(int)
    filtered.rename(columns={'min': 'binMinRequestTime'}, inplace=True)
    filtered.rename(columns={'max': 'binMaxRequestTime'}, inplace=True)
    filtered.to_csv(BINFILENAME, index=False)
    del filtered


if not os.path.isfile(PROPERTIES_FILENAME) or FORCE_GENERATE_PROPERTIES:
    # Calculate object frequency
    print('Calculating Object Frequency')
    objfreqdf = (reqdf['object_ID'].value_counts()).to_frame()
    objfreqdf.rename(columns={'object_ID': 'frequency'}, inplace=True)
    objfreqdf['object_ID'] = objfreqdf.index

    # Calculate object lifespan
    print('Calculating Object LifeSpan & Introduction Day')
    reqdf.sort_values(by=['object_ID'], inplace=True)
    objLifespandf = reqdf.groupby(['object_ID']).agg({'request_time': ['min', 'max']})
    objLifespandf.columns = ['_'.join(col).strip() for col in objLifespandf.columns.values]
    objLifespandf.rename(columns={'request_time_min': 'minRequestTime'}, inplace=True)
    objLifespandf.rename(columns={'request_time_max': 'maxRequestTime'}, inplace=True)
    objLifespandf['object_ID'] = objLifespandf.index
    objLifespandf['lifeSpan'] = (objLifespandf['maxRequestTime'] - objLifespandf['minRequestTime'])/86400
    min_request_time = reqdf['request_time'].min()
    objLifespandf['start_day'] = (objLifespandf['minRequestTime'] - min_request_time) / 86400
    objLifespandf['end_day'] = (objLifespandf['maxRequestTime'] - min_request_time) / 86400
    objLifespandf["start_day"] = objLifespandf["start_day"].apply(int)
    objLifespandf["end_day"] = objLifespandf["end_day"].apply(int)
    objLifespandf.sort_values('start_day', ascending=True, inplace=True)
    objLifespandf.index.names = ['index']

    # Save the properties of the objects
    mergeddf = pd.merge(objfreqdf, objLifespandf, on='object_ID')
    mergeddf = mergeddf[['object_ID', 'frequency', 'lifeSpan', 'minRequestTime', 'maxRequestTime', 'start_day',
                         'end_day']]
    mergeddf.sort_values('start_day', ascending=True, inplace=True)
    mergeddf.to_csv(PROPERTIES_FILENAME, index=False)

    print('Properties File Saved')
    del objfreqdf
    del objLifespandf
    del mergeddf

del reqdf
print('Done')
