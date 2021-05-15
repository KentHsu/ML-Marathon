# Machine learning Marathon 100 days
## Introduction
### Day 1
#### What we should think when we get a data?

1. Why is this quesiton important?
2. Where do data come from?	
	* Data quality is correlated to it's source
3. What are they?
	* Structure data: table
	* Unstructure data: image, video, audio, text
4. What's our goal?
	* [Evaluation metrics] (https://blog.csdn.net/aws3217150/article/details/50479457)

### Day 2
#### Machine learning types
![Machine Learning Type] (Machine_Learning_Type.png)

### Day 3
#### Machine learning project development process
![picture] (ML project process.jpg)

1. Data collection and preprocessing
	* Collection: 
		* What's the data source?
		* What's the data type?
	* Preprocessing:
		* Imputation
		* Outlier handling
		* Normalization

2. Define target and metrics
	* Target/y and Predictor/X
	* Training/Validation/Test datasets
	* Regression/Classification
	* Metrics:
		* Regression: RMSE, MAE, R2
		* Classification: Accuracy, F1 score, ROC, AUC

3. Build model and adjust hyperparameter
	* Regression
	* Tree-based
	* Neutral network
	
4. Deploy model
	* Buildup data collection and preprocessing process
	* Deploy model and output prediction
	* Integrate Frond/Back end

References:  
1. [Google AI Blog] (https://ai.googleblog.com/)
2. [Facebook Research Blog] (https://research.fb.com/blog/)
3. [Apple Machine Learning Journal] (https://machinelearning.apple.com/)
4. [機器之心] (https://www.jiqizhixin.com/)
5. [雷鋒網] (https://www.leiphone.com/category/ai)

### Day 4
#### What's EDA?
* Use statistics and visualization
	* To get data information/structure/characteristic
	* To detect outlier/abnormality
	* To analyze correlation between variables
* Check hypothesis before building up models
* Adjust analysis direction based on EDA result


## Preprocessing
### @ Data collection
### Day 5
#### Buildup DataFrame using Pandas
* Store result in a structured type
* Test code with smal datasets

#### Read data from different data format
* txt

	```python
	with open('example.txt', 'r') as fin:
		data = fin.readlines()
	```
* jpg/png

	```python
	from PIL import Image
	img = Image.read('example.jpg')
	```
	```python
	import skimage.io as skio
	img = skil.imread('example.jpg')
	```
	```python
	import cv2
	img = cv2.imread('example.jpg')
	```
* json

	```python
	import json
	with open('example.json', 'r') as fin:
		data = json.load(fin)
	```
* mat

	```python
	import scipy.io as sio
	data = sio.loadmat('example.mat')
	```
* npy: [Reference] (https://towardsdatascience.com/why-you-should-start-using-npy-file-more-often-df2a13cc0161)

	```python
	import numpy as np
	data = np.load('example.npy')
	```
* pkl

	```python
	import pickle
	with open('example.pkl', 'r') as fin:
		data = pickle.load(fin)
	```

### @ Format adjustment
* Go to Day 13: Operating data on Pandas DataFrame

### Day 6
* Discrete/Continuous data
* float64, int64, object, boolean, string, date
* Label encoding: Ordered categorical dataset

	```python
	
	```
	
* One hot encoding: Unordered categorical dataset

	```python
	
	```
[Referece] (https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621)

### Day 7
* Variable (Feature) types
	* Qualitative (Categorical)
		* Nominally scale
		* Ordinal scalse
	* Quantitative (Numerical)
		* Interval scale
		* Ratio scale
	* Others: Dichotomous, Sequential, Graph

### Day 8
* Statistics
	* Location
		* mean, median, mode
	* Scale
		* min, max, range, quartile, variance, standard deviation
* Visualization
	* Matploblib, Seaborn

	[Reference 1] (http://www.hmwu.idv.tw/web/R_AI_M/AI-M1-hmwu_R_Stat&Prob_v2.pdf), [Reference 2] (https://www.healthknowledge.org.uk/public-health-textbook/research-methods/1b-statistical-methods/statistical-distributions), [Reference 3] (https://amaozhao.gitbooks.io/pandas-notebook/content/pandas%E4%B8%AD%E7%9A%84%E7%BB%98%E5%9B%BE%E5%87%BD%E6%95%B0.html), [Reference 4] (https://chrisalbon.com/python/data_wrangling/pandas_dataframe_descriptive_stats/)

### @ Imputation
* Go to Day 11

### @ Outlier detection
### Day 9
* Root cause
	* unknown value or error 
* How to check?
	* EDA 
* How to handle?
	* drop whole row/column
	* imputation
	* build another row/column to record
* ECDF plot

	[Reference 1] (https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba), [Reference 2] (https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)

### Day 10
Outlier will impact result of feature scaling (standardize/minmax)

* drop it
	* pd.DataFrame.clip(lower, upper) or use masking
* replace it
	* pd.DataFrame.fillna()

### @ Feature scaling

### Day 11
* How to handle outlier?
	* imputation: np.median, np.quantile, np.mean, mode
* Standardization
	* Z-transform: (x - mean(x)) / std(x)
	* Y = 0 ~ 1: (x - min(x)) / (Max(x) - min(x))
	* Y = -1 ~ 1: ((x - min(x)) / (Max(x) - min(x)) - 0.5 ) * 2
	* Y = 0 ~ 1: x / 255 (especially for image processing)

	[Reference] (https://stats.stackexchange.com/questions/189652/is-it-a-good-practice-to-always-scale-normalize-data-for-machine-learning)
	
### Day 12
* Imputation
	* statistics: median, mean, mode
	* specific values
	* values learned from data and predicted by model
* Standardization
	* Standard Scaler: if data comes from normal distribution
	* MinMax Scaler: if data comes from uniform distribution
	* Impact
		* if it's non-tree based model: shows impact
		* if it's tree based model: not impact


	[Reference 1] (https://juejin.im/post/5b5c4e6c6fb9a04f90791e0c), [Reference 2] (https://blog.csdn.net/pipisorry/article/details/52247379)


### Day 13
* Operating data on Pandas: refer to cheat sheet

	[Reference 1] (https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf), [Reference 2] (https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf)
	

## EDA

### Day 14
correlation

### Day 15
correlation

### Day 16
KDE: A nonparameteric method used to plot a probability density function of a random variable.

* property: normalization, symmetry
* Useful kernel: Gaussian, Cosine
	
[Reference 1] (https://python-graph-gallery.com/), [Reference 2] (https://www.r-graph-gallery.com/), [Reference 3] (https://bl.ocks.org/mbostock)

[Reference 4] (https://blog.csdn.net/unixtch/article/details/78556499)

### Day 17
discretizing

* Equal width: pd.cut
* Equal frequency: pd.qcut

### Day 18
discretizing


### Day 19
subplots


### Day 20
* Heatmap
* Pair/Grid plot

[Reference] (https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166)


### Day 21
Just Kaggle experience


## Feature Engineering
### Day 22
Feature engineering is a Fact to Score transformation

[Reference 1] (https://www.zhihu.com/question/29316149), [Reference 2] (https://www.slideshare.net/HJvanVeen/feature-engineering-72376750)


### Day 23
Numeric type - reducing skewness

* log1p transformation: ```np.log1p()```
* sqrt transformation: ```np.sqrt()```
* boxcox transformation ```np.boxcox()```

### Day 24
Categorical type

* Label encoding
	* not deep learning model, especially tree-based model
* One hot encoding
	* deep learning model
	* Used when feature is important and category numbers are small

[Reference] (https://www.twblogs.net/a/5baab6e32b7177781a0e6859?lang=zh-cn)


### Day 25
Categorical type

* Mean encoding
	* If categorical variable shows significant correlation with targets, use **mean of targets** to encode this variable.
	* Smoothing
		* (mean of category * no. of category samples  + grand mean * adjust factor) / (no. of category samples + adjust factor)
	* Fallback
		* overfitting even though we use smoothing


### Day 26
Categorical type

* Count encoding
	* counting mean number of data
	* when mean of target is positive correlated to number of data
	* sklearn count vectorizer in NLP
* Feature hashing
	* used when feature has a lot of category
		* label encoding: order has no meaning
		* one hot encoding: time/space waste
	* Embedding is a better way to solve this problem

	[Reference] (https://blog.csdn.net/laolu1573/article/details/79410187)

### Day 27
DateTime type


### Day 28
Synthesis feature = Feature crosses  
numeric tyep + numeric type

[Reference] (https://segmentfault.com/a/1190000014799038)


### Day 29
Synthesis feature = Feature crosses  
categorical tyep + numeric type  
--> Group by encoding  
Group by encoding doesn't use target value, so it will not cause overfitting like mean encoding

[Reference] (https://zhuanlan.zhihu.com/p/27590154)


### Day 30
Increase features: Feature crosses  
Decrease features: Feature selection

* Filter
	* Correlation filter
* Wrapper
* Embedded 
	* Lasso embedded
	* GDBT embedded: XGBoost

[Reference 1] (https://zhuanlan.zhihu.com/p/32749489), [Reference 2] (https://machine-learning-python.kspax.io/intro-1)

### Day 31

Feature importance in tree-based model:

* weight
* cover
* gain

Sklearn only covers weight but XGBoost covers weight, cover, gain


Permutation importance for non-tree-based model:

change the order of a feature -> check reference 2


[Reference 1] (https://juejin.im/post/5a1f7903f265da431c70144c), [Reference 2] (https://www.kaggle.com/dansbecker/permutation-importance?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights)


### Day 32

Leaf encoding

* Discretize data with leaf node of tree-based model
* Combine with logistic or fractorization machine to improve prediction

[Reference 1] (https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#example-ensemble-plot-feature-transformation-py), [Reference 2] (https://zhuanlan.zhihu.com/p/31734283), [Reference 3] (https://kknews.cc/code/62k4rml.html)


## Model selection
### Day 33
* Define model
* Evaluate model by objective function or loss function
* Tuning hyperparameters: Gradient descent, additive training
* Overfitting
	* increase data
	* decrease model complexity
	* Regularization

[Reference] (http://bangqu.com/yjB839.html)


### @ Introduction
### Day 34
Validation

* split dataset 
	* sklearn.model_selection.train_test_split
* K-fold cross-validation
	* sklearn.model_selection.KFold

### Day 35
Prediction type

* Regression vs Classification
* Multi-class and Multi-label


### Day 36
Evaluation metrics

Regression: MSE, MAE, R2
Classification: accuracy, precision, recall, f1, AUC, ROC, confusion matrix

[Reference 1] (https://www.dataschool.io/roc-curves-and-auc-explained/), [Reference 2] (https://zhuanlan.zhihu.com/p/30721429)


### @ Basic models
### Day 37

Linear regression / Logistic regression

[Reference 1] (hhttps://brohrer.mcknote.com/zh-Hant/how_machine_learning_works/how_linear_regression_works.html), [Reference 2] (https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-3%E8%AC%9B-%E7%B7%9A%E6%80%A7%E5%88%86%E9%A1%9E-%E9%82%8F%E8%BC%AF%E6%96%AF%E5%9B%9E%E6%AD%B8-logistic-regression-%E4%BB%8B%E7%B4%B9-a1a5f47017e5), [Reference 3] (https://www.cnblogs.com/ModifyRong/p/7739955.html)


### Day 38

sklearn implementation: linear/logistic regression

***_WE ARE NOT YET DONE WITH THOSE DATASETS!!!_***

[Reference 1] (https://blog.csdn.net/lc574260570/article/details/82116197), [Reference 2] (https://www.quora.com/What-is-the-difference-between-one-vs-all-binary-logistic-regression-and-multinomial-logistic-regression), [Reference 3] (https://stats.stackexchange.com/questions/120329/what-is-the-difference-between-logistic-and-logit-regression/120364#120364), [Reference 4] (https://github.com/trekhleb/homemade-machine-learning), [Reference 5] (https://dataaspirant.com/2017/05/15/implement-multinomial-logistic-regression-python/)


### Day 39

LASSO / Ridge regression

[Reference 1] (https://www.zhihu.com/question/38121173), [Reference 2] (https://blog.csdn.net/daunxx/article/details/51578787)


### Day 40

sklearn implementation: LASSO/Ridge regression


### @ Tree models
### Day 41

Decision tree

 * information gain
 * entropy
 * gini index
 * feature importance

[Reference] (https://dataaspirant.com/2017/01/30/how-decision-tree-algorithm-works/)

### Day 42

Implement decision tree by sklearn


### Day 43

Random Forest
Ensemble learning: bagging (bootstrap aggregating)

[Reference 1] (https://medium.com/@Synced/how-random-forest-algorithm-works-in-machine-learning-3c0fe15b6674), [Reference 2] (http://hhtucode.blogspot.com/2013/06/ml-random-forest.html)

### Day 44

Implement random forest by sklearn


### Day 45

Gradient Boosting Machine
Ensemble learning: boosting

Modify tree in random forest by improve previous tree's loss with gradient

Bagging create tree by sampling
Boosting create tree additively

Reference 2 is a must-read...

[Reference 1] (https://medium.com/kaggle-blog), [Reference 2] (https://www.youtube.com/watch?v=tH9FH1DH5n0), [Reference 3] (https://explained.ai/gradient-boosting/index.html), [Reference 4] (https://ifun01.com/84A3FW7.html), [Reference 5] (https://www.youtube.com/watch?v=ufHo8vbk6g4), [Reference 6] (https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)

### Day 46

Implement GBM by sklearn

[Reference 1] (https://www.quora.com/Is-multicollinearity-a-problem-with-gradient-boosted-trees), [Reference 2] (https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)

## Parameter tuning

### Day 47

Grid search / Randomized search

[Reference 1] (https://medium.com/rants-on-machine-learning/smarter-parameter-sweeps-or-why-grid-search-is-plain-stupid-c17d97a0e881), [Reference 2] (https://cambridgecoding.wordpress.com/2016/04/03/scanning-hyperspace-how-to-tune-machine-learning-models/), [Reference 3] (https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)


### Day 48

Kaggle basic: sklearn


## Ensemble

### Day 49

Blending

* ensemble in data aspect: Use different data to train the same model
	* bagging: Random forest
	* boosting: GBM, AdaBoost

* ensemble in model, feature aspect: Use different model but the same data
	* blending: voting the result by different model using different weight 
	* stacking


### Day 50

Stacking
Not only blend prediction result but also use the results as new feature


## Unsupervised learning

### @ Clustering

### Day 54
Introduction
	
### Day 55
KMean

### Day 56
silhouette analysis

### Day 57
Hierarchical clustering

### Day 58
2D toy datasets

### @ Dimension Reduction

### Day 59
PCA

### Day 60