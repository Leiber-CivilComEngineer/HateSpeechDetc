# HateSpeechDetction
This the repo is used to perform a dozen of different machine learning models to detect sexist hatespeech on social media. The sorce of sexist hatespeech is from two datasets, "SWSR" a Chinese dataset, "CallMeSexist" a English dataset. The repo also contains some analysis code to perform PCA, TSNE, Topic modeling to further analyze the result retrived by different machine learning models on datasets of different languages.



## Strutures:

### util.py

Helper functions

### load_data

Load csv file, and then tokenise and remove stop words

### imbedding.py

Contains different embedding methods such as tfidf, word2vec, fastest.

### conf.json

Configration file. Used for extension and maintenance.

### result

Saved results

### models

Traditional models and deep model

### data

The sexist hate speech datasets

### analysis

Sentence Bert, Kmeans, and Topic Bert







