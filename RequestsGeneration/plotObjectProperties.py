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
    This code generates the following plots for the analyzed requests file.
    1- {RequestFile}_ActiveObjects: the number of unique active objects requested in each hour
    2- {RequestFile}_Frequency: the frequency distribution of the generated objects
    3- {RequestFile}_Lifespan: the ratio of objects generated for each lifespan value
    4- {RequestFile}_ObjectIntroduction: ratio of objects introduced every day
    5- {RequestFile}_HourlyRequestRatio: ratio to generate object requests per hour (for MediSyn Dataset)


PREPROCESSING_SCRIPTS:
    Need to run the requestAnalysis.py script before running the plotObjectProperties.py script
    requestAnalysis.py: analyzes the generated requests and generates _bin, _properties files used here.


INPUT:
    The input directory is '../Datasets':
    1- REQUESTFILENAME: The request file to be analyzed @Line 36
    2- PLOT_EXT: the extension of the generated plots, by default it is pdf (another alternative is png)


OUTPUT:
    The output plots are generated in '../Datasets' directory.
"""

from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import collections


REQDIR = '../Datasets/'
REQUESTFILENAME = 'mediSynDataset_x2_O3488.csv'  #'syntheticDataset_O50.csv'
BINFILENAME = REQDIR + REQUESTFILENAME[:-4] + '_bins.csv'
PROPERTIES_FILENAME = REQDIR + REQUESTFILENAME[:-4] + '_properties.csv'
HOURLY_REQUEST_RATIO_FILE = 'hourly_request_ratio.csv'

PLOT_PATH_PREFIX = '{}{}_'.format(REQDIR, REQUESTFILENAME[:-4])
PLOT_EXT = 'pdf'


"""####################  Plotting number of active objects for different bins  ######################################"""
print('Plotting Number of Active Objects ...')
bindf = pd.read_csv(BINFILENAME)
bindf['binID'] = bindf['binID'] - bindf.iloc[0]['binID']
bindf['binID'] = bindf['binID'].astype('int')
plt.close('all')
plt.figure(figsize=(3, 2))
bindf.plot('binID', 'uniqueObjNum', legend=None, color='blue')
plt.xlabel('Hourly Bins')
plt.ylabel('Number of Active Objects')
plt.title('Number of Active Objects per Hour')
plt.grid()
plt.tight_layout()
plt.savefig('{}ActiveObjects.{}'.format(PLOT_PATH_PREFIX, PLOT_EXT), format=PLOT_EXT, facecolor='white', dpi=300)


"""####################  Plotting frequency of each object  #########################################################"""
print('Plotting Object Frequency ...')
objdf = pd.read_csv(PROPERTIES_FILENAME)
objdf.sort_values('frequency', ascending=False, inplace=True)
plt.close('all')
plt.figure(figsize=(4, 3))
plt.loglog(range(1, len(objdf.frequency) + 1), objdf.frequency, color='blue')
plt.yscale('log')
plt.xlabel('Object Rank')
plt.ylabel('Object Frequency')
plt.title('Distribution of Object Frequency')
plt.grid()
plt.tight_layout()
plt.savefig('{}Frequency.{}'.format(PLOT_PATH_PREFIX, PLOT_EXT), format=PLOT_EXT, facecolor='white', dpi=300)


"""####################  Plotting ratio of objects for each lifespan value  #########################################"""
print('Plotting Object LifeSpan ...')
plt.close('all')
plt.figure(figsize=(4, 3))
objdf.sort_values('lifeSpan', ascending=True, inplace=True)
lifeSpans = objdf["lifeSpan"].apply(int)
max_days = max(lifeSpans.unique())
days = lifeSpans.unique()
counts = collections.Counter(lifeSpans)
plt.grid(zorder=0)
if len(days) <= 5:
    fig, ax = plt.subplots(1, 1)
    ax.bar(days, [float(counts[key]) / objdf.shape[0] for key in counts], width=0.2, zorder=3, color='blue')
    ax.set_xlim(0, max_days)
else:
    plt.bar(days, [float(counts[key]) / objdf.shape[0] for key in counts], zorder=3, color='blue')
plt.xlabel('Object LifeSpan in days')
plt.ylabel('Ratio (over entire object population)')
plt.title('Ratio of Object LifeSpan')
plt.tight_layout()
plt.savefig('{}Lifespan.{}'.format(PLOT_PATH_PREFIX, PLOT_EXT), format=PLOT_EXT, facecolor='white', dpi=300)


"""####################  Plotting ratio of objects introduced each day  #############################################"""
print('Plotting Ratio of Objects introduced per day ...')
plt.close('all')
plt.figure(figsize=(4, 3))
plt.grid(zorder=0)
objdf.sort_values('start_day', ascending=True, inplace=True)
start_days = objdf["start_day"].apply(int)
days = start_days.unique()
counts = collections.Counter(start_days)
if len(days) <= 5:
    fig, ax = plt.subplots(1, 1)
    ax.bar(days, [float(counts[key]) / objdf.shape[0] for key in counts], width=0.2, zorder=3, color='blue')
    ax.set_xlim(0, max_days)
else:
    plt.bar(days, [float(counts[key]) / objdf.shape[0] for key in counts], zorder=3, color='blue')
plt.title('Ratio of New Objects Introduced per Day')
plt.xlabel('Simulation Time in days')
plt.ylabel('Ratio (over entire object population)')
plt.tight_layout()
plt.savefig('{}ObjectIntroduction.{}'.format(PLOT_PATH_PREFIX, PLOT_EXT), format=PLOT_EXT, facecolor='white', dpi=300)

del bindf
del objdf


"""####################  Plotting hourly request ratio of objects  ##################################################"""
print('Plotting Hourly request ratio ...')
if 'mediSynDataset' in REQUESTFILENAME:
    hour_date_df = pd.read_csv(HOURLY_REQUEST_RATIO_FILE, header=None, skiprows=[0, 0], names=['hour', 'ratio'])
    plt.close('all')
    plt.figure(figsize=(4, 3))
    plt.bar(hour_date_df.hour, hour_date_df.ratio, color='blue')
    plt.title('Hourly Access Ratio')
    plt.ylabel('Access Ratio')
    plt.xlabel('Hour')
    plt.tight_layout()
    plt.savefig('{}HourlyRequestRatio.{}'.format(PLOT_PATH_PREFIX, PLOT_EXT), format=PLOT_EXT,
                facecolor='white', dpi=300)
    del hour_date_df


print('Done')
