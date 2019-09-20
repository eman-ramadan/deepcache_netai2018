## [DeepCache: A Deep Learning Based Framework For Content Caching](https://dl.acm.org/citation.cfm?id=3229555)
### By: Arvind Narayanan, Saurabh Verma, Eman Ramadan, Pariya Babaie, and Zhi-Li Zhang
In this paper, we present DEEPCACHE a novel Framework for content caching, which can significantly boost cache performance. Our Framework is based on powerful deep recurrent neural network models. It comprises of two main components: i) Object Characteristics Predictor, which builds upon deep LSTM Encoder-Decoder model to predict the future characteristics of an object (such as object popularity) -- to the best of our knowledge, we are the first to propose LSTM Encoder-Decoder model for content caching; ii) a caching policy component, which accounts for predicted information of objects to make smart caching decisions. In our thorough experiments, we show that applying DEEPCACHE Framework to existing cache policies, such as LRU and k-LRU, significantly boosts the number of cache hits.

DeepCache is an open source version of the DeepCache platform described in our [NetAI 2018](https://conferences.sigcomm.org/sigcomm/2018/workshop-netaim.html) paper, released under the [BSD](LICENSE) license.

*The available code now ONLY generates the synthetic and MediSyn datasets used in our paper.*

### Requests Generation:
* [`generateSyntheticDataset.py`](RequestsGeneration/generateSyntheticDataset.py): 
This code generates a synthetic dataset which 
has multiple sessions where the object popularity and the request 
interarrival distribution changes from one session to another.


* [`generateMediSynDataset.py`](RequestsGeneration/generateMediSynDataset.py): This code is based on Tang, Wenting, et al. 
["MediSyn: A synthetic streaming media service workload generator."](https://dl.acm.org/citation.cfm?id=776327)
Proceedings of the 13th international workshop on Network and operating systems
support for digital audio and video. ACM, 2003.

	The variables are mostly based on the paper, but we did some changes to generate
	requests which fit our need as shown in the request analysis plots. In addition 
	to object popularity, and request interarrival distribution, this dataset
	simulates real traffic by:
	1. having new objects introduced at different times
	2. objects have different types with variable lifespans
	3. requests for each object are generated based on an hourly request ratio



### Requests Analysis:
* [`requestAnalysis.py`](RequestsGeneration/requestAnalysis.py): analyzes the generated requests to generate the number 
of active objects, object frequency, object lifespan, object introduction day.
* [`plotObjectProperties.py`](RequestsGeneration/plotObjectProperties.py): plots the previous data about object properties for
 the generated dataset.



### Sample Datasets with Analysis Plots:
[Synthetic Dataset Sample](https://drive.google.com/open?id=1TItVsjbwQ3gZjlqjixO-ywBG9BJ3b1B_)

[MediSyn Dataset Sample](https://drive.google.com/open?id=1S6A_oNDFSnH0pvSzKvhoM6lHSA0UsKzJ)



### Citing DeepCache:
If you use DeepCache in your work, please cite our paper:
```
@inproceedings{Narayanan:2018:DDL:3229543.3229555,
	 author = {Narayanan, Arvind and Verma, Saurabh and Ramadan, Eman and Babaie, Pariya and Zhang, Zhi-Li},
	 title = {DeepCache: A Deep Learning Based Framework For Content Caching},
	 booktitle = {Proceedings of the 2018 Workshop on Network Meets AI \& ML},
	 series = {NetAI'18},
	 year = {2018},
	 isbn = {978-1-4503-5911-5},
	 location = {Budapest, Hungary},
	 pages = {48--53},
	 numpages = {6},
	 url = {http://doi.acm.org/10.1145/3229543.3229555},
	 doi = {10.1145/3229543.3229555},
	 acmid = {3229555},
	 publisher = {ACM},
}
```
