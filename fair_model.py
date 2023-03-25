import pandas as pd
import cornac
import mip
import numpy as np
from tqdm import tqdm
from mip import Model, BINARY, xsum, maximize, MAXIMIZE
from cornac.data import Reader
from cornac.eval_methods import RatioSplit, BaseMethod
from cornac.metrics import Precision, Recall, NDCG
from cornac.models import VAECF, WMF, NeuMF, MostPop, BPR
import collections

items_to_screen = 50
result = 10
ml20m = pd.read_csv(filepath_or_buffer='~/ml/ml100k.csv')
#remove timestep info
del ml20m["timestamp"]
#remove users having made less than 5 ratings and items which has less than 5 ratings
user_inter, item_inter = 1, 1
while user_inter != 0 or item_inter != 0 :
    uid_value_counts = ml20m["userId"].value_counts()
    user_inter = uid_value_counts[uid_value_counts < 5].count()
    iid_value_counts = ml20m["movieId"].value_counts()
    item_inter = iid_value_counts[iid_value_counts < 5].count()
    print(f"user with interactions < 5 : {user_inter} item with interations < 5 : {item_inter}")
    uid_value_counts = uid_value_counts.to_dict()
    ml20m["count"] = ml20m["userId"].map(uid_value_counts)
    ml20m = ml20m[ml20m["count"] > 4]
    iid_value_counts = iid_value_counts.to_dict()
    ml20m["count"] = ml20m["movieId"].map(iid_value_counts)
    ml20m = ml20m[ml20m["count"] > 4]
del ml20m["count"]

print(ml20m.describe())
msk = np.random.rand(len(ml20m)) < 0.8
train = ml20m[msk]
test = ml20m[~msk]

train.to_csv('~/ml/trainset.csv', index=False)
test.to_csv('~/ml/testset.csv', index=False)
print(train.describe())

#write the cleaned data back to data file
train.to_csv(path_or_buf='~/ml/trainset.csv', index=False)
test.to_csv(path_or_buf='~/ml/testset.csv', index=False)
reader = Reader()
train_data = reader.read(fpath=f'/home/pyw/ml/trainset.csv', fmt="UIR", sep=',', skip_lines=1)
test_data = reader.read(fpath=f'/home/pyw/ml/testset.csv', fmt="UIR", sep=',', skip_lines=1)
#set the metrics to be precision, recall, nDCG and accuracy
eval_method = BaseMethod.from_splits(train_data=train_data, test_data=test_data, rating_threshold=2, exclude_unknowns=False, seed=123)
user_num = eval_method.train_set.total_users
item_num = eval_method.train_set.total_items


def cornac_exp():
    precision = Precision(k=10)
    recall = Recall(k=10)
    ndcg_10 = NDCG(k = 10)
    #set the comparision models to be most popular, Neural Matrix Factorizaiton, Wide Matrix Factorization and Variational AutoEncoder for Collaborative Filtering
    #selcet one model at a time is recommended for memrory size may be limited
    models = [#MostPop()]
          BPR(k=50, max_iter=200, learning_rate=0.001, lambda_reg=0.001, verbose=True)]
          #NeuMF(num_factors=9, layers=[32, 16, 8], act_fn="tanh", num_epochs=5, num_neg=3, batch_size=256, lr=0.001, seed=42, verbose=True)]
          #WMF(k=50, max_iter=50, learning_rate=0.001, lambda_u=0.01, lambda_v=0.01, verbose=True, seed=123)]
          #VAECF(k=50,batch_size=100,seed=123,use_gpu=True)]
          
    #get the experiment loaded on defined dataset
    exp = cornac.Experiment(eval_method=eval_method, models=models, metrics=[precision, recall, ndcg_10])
    exp.run()
    return exp

#get a matrix of whether a uid is on the short list
def save_short_uid(dataset = train, datamap = eval_method.train_set.uid_map):
    uid_value_counts = dataset["userId"].value_counts()
    mid = round(uid_value_counts.mean())
    um = np.zeros((user_num, 2))
    for uid in pd.unique(dataset["userId"]):
        uidx = datamap[str(uid)]
        if uid_value_counts[uid] > mid:
            um[uidx][0] = 1
        else:
            um[uidx][1] = 1
    return um

#get a set of items with more than average remarks
def save_short_iid(dataset = train, datamap = eval_method.train_set.iid_map):
    iid_value_counts = dataset["movieId"].value_counts()
    short_uid = set()
    mid = round(iid_value_counts.mean())
    for iid in pd.unique(dataset["movieId"]):
        if iid_value_counts[iid] > mid:
            iidx = datamap[str(iid)]
            short_uid.add(iidx)
    return short_uid

#get 2 dights ratio of head / long tail users
def short_uid_ratio(dataset = ml20m):
    return round(len(save_short_uid(dataset=dataset)) / dataset["userId"].value_counts().count(), 2)

#get 2 dights ratio of head / long tail items
def short_iid_ratio(dataset = ml20m):
    return round(len(save_short_iid(dataset=dataset)) / dataset["movieId"].value_counts().count(), 2)

#get a matrix of each users' top 50 recommended item
def load_scores():
    exp = cornac_exp()
    models = exp.models
    S = np.zeros((user_num, items_to_screen))
    R = np.zeros((user_num, items_to_screen))
    for model in models:
        print(model.name)
        for uid in tqdm(range(user_num)):
            rank, scores = model.rank(uid)
            S[uid] = np.array(list(scores))[:items_to_screen]
            R[uid] = np.array(list(rank))[:items_to_screen]
    unique_item = set()
    for i in tqdm(range(uid)):
        for j in range(result):
            unique_item.add(R[i][j])
    exposure = 1.0 * len(unique_item) / item_num
    print(round(exposure, 4))
    return S, R

#get a binary matrix shows whether a selected item belongs to long tail item group
def selected_item_matrix(R, short_iid_list):
    im = np.zeros((user_num, items_to_screen, 2))
    for uid in tqdm(range(user_num)):
        for iid in range(items_to_screen):
            if R[uid][iid] in short_iid_list:
                im[uid][iid][1] = 1
            else:
                im[uid][iid][0] = 1
    return im

#make a dictionary with user indices as keys and their choices in trainset as values
def ui_dic(trainset=eval_method.train_set):
    dic = collections.defaultdict(set)
    iter = trainset.uir_iter()
    for i in tqdm(range(trainset.num_ratings)):
        uid, iid, _ = next(iter)
        dic[uid[0]].add(iid[0])
    return dic

#get a binary matrix shows whether a recommended item was actually selected
def right_ranked_matrix(R, dic):
    um = np.zeros((user_num, items_to_screen))
    for uid in tqdm(range(user_num)):
        for iid in range(items_to_screen):
            if R[uid][iid] in dic[uid]:
                um[uid][iid] = 1
    return um

#count the number of times when items on the short list being choosed
def item_portion(dic, full_length = len(train)):
    iid_list = save_short_iid()
    count = 0
    for uid in tqdm(range(user_num)):
        for iid in dic[uid]:
            if iid in iid_list:
                count += 1
    return full_length - count, count

def precisionk(actual, predicted):
  return 1.0 * len(set(actual) & set(predicted)) / len(predicted)

def recallk(actual, predicted):
  return 1.0 * len(set(actual) & set(predicted)) / len(actual)

def ndcgk(actual, predicted):
  idcg = 1.0
  dcg = 1.0 if predicted[0] in actual else 0.0
  for i, p in enumerate(predicted[1:]):
    if p in actual:
      dcg += 1.0 / np.log(i+2)
    idcg += 1.0 / np.log(i+2)
  return dcg / idcg
        
S, R = load_scores()
U = save_short_uid()
dic = ui_dic()
T = right_ranked_matrix(R, dic)
portion = item_portion(dic)
I = selected_item_matrix(R, save_short_iid())

#solve the mixed integer problem of which 10 item among 50 shoule be elected
def fairness_op(uep = 0.00005, iep = 0.00005):
    model = Model(name="fair", sense=MAXIMIZE)
    W = [[model.add_var(var_type=BINARY) for i in np.zeros(items_to_screen)] for j in np.zeros(user_num)]
    
    user_cons = [model.add_var() for i in np.zeros(2)]
    item_cons = [model.add_var() for i in np.zeros(2)]
    tp = [model.add_var() for i in range(user_num)]
    #model.objective = maximize(xsum(S[i][j] * W[i][j] for i in range(user_num) for j in range(items_to_screen)))
    model.objective = maximize(xsum(S[i][j] * W[i][j] for i in range(user_num) for j in range(items_to_screen)) - uep * (user_cons[0] - user_cons[1]) - iep * (item_cons[0] - item_cons[1]))
    for i in range(user_num):
        model += xsum(W[i][j] for j in range(items_to_screen)) == result
    for i in range(user_num):
        model += tp[i] == xsum(W[i][j] * T[i][j] for j in range(items_to_screen))
    for k in range(2):
        model += item_cons[k] == xsum(W[i][j] * I[i][j][k] for i in range(user_num) for j in range(items_to_screen)) 
    for k in range(2):
        model += user_cons[k] == xsum(tp[i] * U[i][k] for i in range(user_num))
    model.optimize()
    return W

def model_metric(W):
    predicted = list()
    NDCG_all = list()
    PRE_all = list()
    REC_all = list()
    unique_item = set()
    for uid in tqdm(range(user_num)):
        if uid in dic.keys():
            for iid in range(items_to_screen):
                if W[uid][iid] == 1:
                    predicted.append(R[uid][iid])
                    unique_item.add(R[uid][iid])
            ndcg_one = ndcgk(dic[uid], predicted)
            pre_one = precisionk(dic[uid], predicted)
            rec_one = recallk(dic[uid], predicted)
            NDCG_all.append(ndcg_one)
            PRE_all.append(pre_one)
            REC_all.append(rec_one)
            predicted.clear()
    exposure = 1.0 * len(unique_item) / item_num
    print(round(exposure, 4))
    print(round(np.mean(NDCG_all), 4))
    print(round(np.mean(PRE_all), 4))
    print(round(np.mean(REC_all), 4))


W = fairness_op()
model_metric(W)
print("over")