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


@author: Eman Ramadan (eman@cs.umn.edu) & Pariya Babaie (babai008@umn.edu)


DESCRIPTION:
    This code generates a synthetic dataset which has multiple sessions in which the object popularity and the request
    interarrival distribution changes from one session to another.


INPUT:
    - alphaSet: values for object popularity Zipf distribution for each session.
    - NUM_OF_OBJECTS: number of objects for request generation per session, Object_ID from 1 to NUM_OF_OBJECTS
    - NUM_OF_REQUESTS_PER_SESSION: number of requests per session
    Total number of requests generated approx. equals num_of_sessions * NUM_OF_REQUESTS_PER_SESSION
    These inputs are hard coded, so no need to send any arguments to the script, you can run it directly.
    You can modify these parameters in the {initialization} section @Line 45


FLOW:
    1- For each value of alpha in the alpha set, 'NUM_OF_REQUESTS_PER_SESSION' requests are generated in each session.
    2- The rank of objects (and hence their popularity) are permutated in each session which means an object which is
       popular in the first session may have a different popularity in the second session and so on. The interarrival
       distribution also varies from one session to the other.
    3- At each session a set of requests for each object are generated and are appended to the
       previously generated requests.
    4- All requests are stored in a data frame which contains object IDs to be requested and their request request_time,
       sorted by the request_time.


OUTPUT:
    The output is generated in '../Datasets' directory with the name:
     'syntheticDataset_O{NUM_OF_OBJECTS}.csv'
    the requests file contains object_IDs to be requested and their request_time.

"""

from __future__ import print_function
import random
import math
import os
import numpy as np
import pandas as pd


""""###########################  initialization  ####################################################################"""
alphaSet = [0.8, 1, 0.5, 0.7, 1.2, 0.6]
NUM_OF_OBJECTS = 50
NUM_OF_REQUESTS_PER_SESSION = 50000

OUTPUTDIR = '../Datasets'
OUTPUTFILENAME = '{}/syntheticDataset_O{}.csv'.format(OUTPUTDIR, NUM_OF_OBJECTS)
if not os.path.isdir(OUTPUTDIR):
    os.mkdir(OUTPUTDIR)


def generate_requests():
    print('Generating Requests for Dataset ...')
    for i in range(len(alphaSet)):
        print('Processing Session Num {} out of {} Sessions'.format(i + 1, len(alphaSet)))
        alpha = alphaSet[i]

        # selecting the distribution randomly for this session
        if random.randint(1, 2) % 2 == 0:
            dist = 'P'    # Poisson distribution
        else:
            dist = 'R'    # Pareto distribution

        if i != 0:
            # New Rank permutation used to change objects rank from one session to another
            perm = np.random.permutation(NUM_OF_OBJECTS) + 1

        print('Alpha = {}, distribution is {}'.format(alpha, 'Poisson' if dist == 'P' else 'Pareto'))
        requests = generate_session_requests(alpha, dist)     # returned requests are object_ID,request_time

        df = pd.DataFrame(requests, columns=['requests'])
        # split the requests into two columns object_ID and request_time
        parts = df["requests"].str.split(",", expand=True)

        # making separate object id column from new data frame
        df["object_ID"] = parts[0]
        df["object_ID"] = df["object_ID"].astype(int)

        # making separate request timestamp column from new data frame
        df["request_time"] = parts[1]
        df["request_time"] = df["request_time"].astype(float)

        if i == 0:
            requestsdf = df[['object_ID', 'request_time']].copy(deep=True)
        else:
            # The following code is used to permutate the rank of each object in the current session
            for cur_obj_id in df.groupby('object_ID').groups.keys():  # cur_obj_id: from 1 to NUM_OF_OBJECTS
                new_obj_id = perm[cur_obj_id - 1]
                # replace the cur_obj_id with the new_obj_id
                new_obj_id_col = np.full((len(df.groupby('object_ID').groups[cur_obj_id])), new_obj_id)

                # adjust the generated request_times to start from the last request_time of the new_obj_id from
                # the previous session
                last_req_time = lastReqs.loc[lastReqs['object_ID'] == cur_obj_id]['request_time_max'].tolist()[0]
                new_request_times_col = df.loc[df['object_ID'] == cur_obj_id, ['request_time']] + last_req_time
                new_request_times_col = new_request_times_col['request_time'].tolist()

                # create a tmp data frame for this object
                tmp_df = pd.DataFrame({'object_ID': new_obj_id_col, 'request_time': new_request_times_col})

                # merge the tmp data frame for this object with existing requests
                requestsdf = requestsdf.append(tmp_df, ignore_index=True)

        # get the last request_time for each object_id to be used for the next session
        lastReqs = requestsdf.groupby(['object_ID']).agg({'request_time': ['max']})
        # flatten the lastReqs column names after grouping
        lastReqs.columns = ['_'.join(col).strip() for col in lastReqs.columns.values]
        lastReqs['object_ID'] = lastReqs.index
        # now lastReqs data frame has each object_ID and its request_time_max (when it was last requested)

    # sort all requests for all objects by request_time
    requestsdf.sort_values('request_time', ascending=True, inplace=True)

    f = open(OUTPUTFILENAME, 'w')
    requestsdf.to_csv(f, header=True, index=False)
    f.close()
    print('Synthetic Dataset Saved to Output file: {}'.format(OUTPUTFILENAME))
    del requestsdf


"""###########################  Object Popularity  ##################################################################"""
def generate_object_popularity_zipf(zipalpha):
    """
    Generate Zipf distribution for object popularity
    """
    N = NUM_OF_OBJECTS
    denom = 0.0
    for i in range(N):
        denom += 1.0/pow((i+1), zipalpha)
    objects_zipf_pdf = []
    for i in range(N):
        item = 1.0/pow((i+1), zipalpha)
        item /= denom
        con_index = i+1
        objects_zipf_pdf.append((con_index, item))
    objects_zipf_pdf = sorted(objects_zipf_pdf, key=lambda a: a[0])
    return objects_zipf_pdf


"""###########################  Session Requests  ###################################################################"""
def generate_session_requests(zipalpha, distr):
    """
    Generate interarrival times following either the poisson or pareto distributions
    """
    # For each object generate lambda_i * num_of_requests according to their interarrival distribution
    # Then merge these requests for all objects with respect to their request time

    requests = []

    # generating object popularity
    objects_zipf_pdf = generate_object_popularity_zipf(zipalpha)

    # Calculating the maximum simulation time by generating the required number of requests for the least popular object
    simulation_time_end = 0
    N = NUM_OF_OBJECTS - 1
    reqs_N = int(math.ceil(objects_zipf_pdf[N][1] * NUM_OF_REQUESTS_PER_SESSION))
    ctr = 0
    cur_t = 0
    while ctr < reqs_N:
        rand = np.random.uniform()
        if distr == 'P':
            t = generate_poisson_distribution_from_CDF(rand, objects_zipf_pdf[N][1])
        elif distr == 'R':
            t = generate_pareto_distribution_from_CDF(rand, objects_zipf_pdf[N][1])
        ctr += 1
        cur_t += t
        simulation_time_end = cur_t
        req_str = str(N+1) + ',' + str(cur_t)
        requests.append(req_str)

    # generate the requests for the other N-1 objects, and stop when the time exceeds the simulation time
    for i in range(NUM_OF_OBJECTS - 1):
        cur_t = 0
        while cur_t < simulation_time_end:
            rand = np.random.uniform()
            if distr == 'P':
                t = generate_poisson_distribution_from_CDF(rand, objects_zipf_pdf[i][1])
            elif distr == 'R':
                t = generate_pareto_distribution_from_CDF(rand, objects_zipf_pdf[i][1])
            cur_t += t
            req_str = str(i+1) + ',' + str(cur_t)
            requests.append(req_str)

    requests = sorted(requests, key=lambda a: float(a.split(',')[1]))
    return requests


def generate_poisson_distribution_from_CDF(rand, lambda_poisson):
    """
    Generate numbers following Poisson distribution using its CDF and a random number uniformly generated
    """
    return -1 * (math.log(1-rand))/lambda_poisson


def generate_pareto_distribution_from_CDF(rand, lambda_pareto):
    """
    Generate numbers following Pareto distribution using its CDF and a random number uniformly generated
    """
    # We use beta = 2 (Pareto Parameter from distribution)
    return (1/math.sqrt(1-rand) - 1)/lambda_pareto


"""##################################################################################################################"""


def main():
    generate_requests()

if __name__ == "__main__": main()
