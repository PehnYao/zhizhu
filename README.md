# zhizhu
This file is wirte for better performance & fairness of existing recommender systems through a re-ranking approach
To user it the following python packages has to be installed in advance
  mip,cornac,tensorflow 1.x, pandas, and pytorch
The data used in code is movieLence 100k dataset, you may change to any other dataset with infomation of users, items and ratings
Please note the file path is hard coded
You may obeserve its performance under different recommeder models by those in line 62-65 or freely adding other models which cornac supports
A touch of test is placed here
    Dataset: ML-100K
  NDCG  Precision Recall  Exposure
  0.1408  0.1215  0.1075  0.1105  VAECF
  0.2832  0.4586  0.4202  0.3741  VAECF with Re-ranking
