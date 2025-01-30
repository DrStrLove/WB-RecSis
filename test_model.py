#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import my_logger
import os
import re
import json
import joblib
import zipfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
from sklearn.metrics import roc_auc_score
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import (
    ImplicitItemKNNWrapperModel,
    ImplicitALSWrapperModel,
    PopularModel
)
from rectools.metrics import (
    Recall, HitRate, Precision, MAP, calc_metrics
)
from rectools.models.serialization import load_model
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import (
    ItemItemRecommender, TFIDFRecommender, BM25Recommender
)


# In[ ]:


my_logger.logger.info("Загрузка тестовых данных.")

file_test_w1 = "warm_test.parquet"
file_test_c1 = "cold_test.parquet"
train_file = "train.parquet"
file_fch = "features.parquet"

test_data_cold = pd.read_parquet(file_test_c1)
test_data_warm = pd.read_parquet(file_test_w1)
history = pd.read_parquet(train_file)
fch_df = pd.read_parquet(file_fch)

my_logger.logger.info("Загрузка тестовых данных успешна.")


# In[ ]:


history_dataset = Dataset.construct(history)

test_users = test_data_warm["user_id"].unique()


# In[ ]:


my_logger.logger.info("Семплирование тестовых данных.")

unique_users = test_data_warm['user_id'].unique()
num_to_remove = int(0.4 * len(unique_users))
users_to_remove = np.random.choice(unique_users, size=num_to_remove, replace=False)
test_data_warm = test_data_warm[~test_data_warm['user_id'].isin(users_to_remove)]


# In[ ]:


unique_users = test_data_cold['user_id'].unique()
num_to_remove = int(0.4 * len(unique_users))
users_to_remove = np.random.choice(unique_users, size=num_to_remove, replace=False)
test_data_cold = test_data_cold[~test_data_cold['user_id'].isin(users_to_remove)]

my_logger.logger.info("Семплирование тестовых данных успешно.")


# In[ ]:


my_logger.logger.info("Загрузка моделей и получение предсказаний.")

knn_model = load_model("knn_model.pkl")
als_model = load_model("als_model.pkl")
popular_model = load_model("popular_model.pkl")

rec_knn_w = knn_model.recommend(
    users=test_users,
    dataset=history_dataset,
    k=50,
    filter_viewed=True
)
my_logger.logger.info("Сгенрирован rec_knn_w")

rec_als_w = als_model.recommend(
    users=test_users,
    dataset=history_dataset,
    k=50,
    filter_viewed=True
)
my_logger.logger.info("Сгенрирован rec_als_w")

rec_pop_w = popular_model.recommend(
    users=test_users,
    dataset=history_dataset,
    k=50,
    filter_viewed=True
)
my_logger.logger.info("Сгенрирован rec_pop_w")

rec_pop_c = popular_model.recommend(
    users=test_users,
    dataset=history_dataset,
    k=50,
    filter_viewed=True
)
my_logger.logger.info("Сгенрирован rec_pop_c")


# In[ ]:


rec_als_w.to_parquet("rec_test_als.parquet", index=False)
my_logger.logger.info("Сохранены предсказания rec_test_als.parquet")

rec_knn_w.to_parquet("rec_test_knn.parquet", index=False)
my_logger.logger.info("Сохранены предсказания rec_test_knn.parquet")

pd.concat([rec_pop_w, rec_pop_c], ignore_index=True).to_parquet("rec_test_pop.parquet", index=False)
my_logger.logger.info("Сохранены предсказания rec_test_pop.parquet")


# In[ ]:


my_logger.logger.info("Загрузка тестовых предсказаний первого уровня")

file_rec_1 = 'rec_test_als.parquet'
file_rec_2 = 'rec_test_knn.parquet'
file_rec_4 = 'rec_test_pop.parquet'


# In[ ]:


my_logger.logger.info("Объединение тестовых предсказаний первого уровня.")

rec_als = pd.read_parquet(file_rec_1)
rec_cos = pd.read_parquet(file_rec_2)
rec_pop = pd.read_parquet(file_rec_4)

rec_als = rec_als.rename(columns={"score": "als_score", "rank": "als_rank"})
rec_cos = rec_cos.rename(columns={"score": "cos_score", "rank": "cos_rank"})
rec_pop = rec_pop.rename(columns={"score": "pop_score", "rank": "pop_rank"})

bst_rec = rec_pop
bst_rec = bst_rec.merge(rec_als, on=["user_id", "item_id"], how="outer")
bst_rec = bst_rec.merge(rec_cos, on=["user_id", "item_id"], how="outer")

bst_rec = bst_rec.fillna(0)


# In[ ]:


my_logger.logger.info("Добавление фичей к тестовым предсказаниям первого уровня")

test_data = pd.concat([test_data_cold, test_data_warm], ignore_index=True)

df_target = bst_rec[['user_id', 'item_id']].merge(
    test_data[['user_id', 'item_id']],
    on=['user_id', 'item_id'],
    how='left',
    indicator=True
)

bst_rec['target'] = (df_target['_merge'] == 'both').astype(int)

del df_target


nique_items = history['item_id'].unique()
item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
idx_to_item = {idx: item for item, idx in item_to_idx.items()}

user_to_idx = {user: idx for idx, user in enumerate(history['user_id'].unique())}
rows = history['user_id'].map(user_to_idx).values
cols = history['item_id'].map(item_to_idx).values
data = np.ones(len(history))
user_item_matrix = coo_matrix((data, (rows, cols)), shape=(len(user_to_idx), len(item_to_idx))).tocsr()

cooccurrence_matrix = user_item_matrix.T.dot(user_item_matrix)

item_frequencies = np.array(user_item_matrix.sum(axis=0)).flatten()

row_indices, col_indices = cooccurrence_matrix.nonzero()
values = []
for i, (row, col) in tqdm(enumerate(zip(row_indices, col_indices)), total=len(row_indices)):
    values.append(
        cooccurrence_matrix[row, col] / np.sqrt(item_frequencies[row] * item_frequencies[col])
    )
cooccurrence_normalized = coo_matrix((values, (row_indices, col_indices)), shape=cooccurrence_matrix.shape).tocsr()

def calculate_mean_normalized_cooccurrence(user_id, candidate_item):
    if user_id not in user_to_idx or candidate_item not in item_to_idx:
        return 0
    user_idx = user_to_idx[user_id]
    candidate_idx = item_to_idx[candidate_item]
    user_items = user_item_matrix[user_idx].indices
    if len(user_items) == 0:
        return 0
    scores = cooccurrence_normalized[candidate_idx, user_items].toarray().flatten()
    return scores.mean() if len(scores) > 0 else 0

bst_rec['m_n_co'] = [
    0 if row['user_id'] not in user_to_idx else calculate_mean_normalized_cooccurrence(row['user_id'], row['item_id'])
    for _, row in tqdm(bst_rec.iterrows(), total=len(bst_rec))
]

item_popularity = history.groupby('item_id')['user_id'].nunique().reset_index()
item_popularity.rename(columns={'user_id': 'it_pop'}, inplace=True)

bst_rec = bst_rec.merge(item_popularity, on='item_id', how='left')

bst_rec['it_pop'] = bst_rec['it_pop'].fillna(0).astype(int)

bst_rec = bst_rec.merge(
    fch_df,
    on='item_id',
    how='left'
)

bst_rec.fillna(0, inplace=True)


# In[ ]:


my_logger.logger.info("Ранжирование тестовых предсказаний на втором уровне")

excluded_cols = ['user_id', 'item_id', 'target']
feature_cols = [c for c in bst_rec.columns if c not in excluded_cols]

catboost_model_path = "catboost_model.cbm"
model = joblib.load(catboost_model_path)

bst_rec['score'] = model.predict_proba(bst_rec[feature_cols])[:, 1]

bst_rec['rank'] = bst_rec.groupby('user_id')['score'].rank(
    method='first', ascending=False
).astype(int)

my_logger.logger.info("Подготовка тестовых переранжированных предсказаний к выводу метрик")

columns_to_keep = ['user_id', 'item_id', 'score', 'rank']
bst_rec.drop(columns=[col for col in bst_rec.columns if col not in columns_to_keep],
             inplace=True)

bst_rec.sort_values(['user_id', 'rank'], inplace=True)


# In[ ]:


my_logger.logger.info("Выввод метрик.")

kb = [10, 30, 50, 100]
metrics = {}

for k in kb:
    recall_metric = Recall(k=k)
    hitrate_metric = HitRate(k=k)
    precision_metric = Precision(k=k)
    map_metric = MAP(k=k)

    metrics[f"recall@{k}"] = recall_metric.calc(
        reco=bst_rec,
        interactions=test_data
    )
    metrics[f"hitrate@{k}"] = hitrate_metric.calc(
        reco=bst_rec,
        interactions=test_data
    )
    metrics[f"precision@{k}"] = precision_metric.calc(
        reco=bst_rec,
        interactions=test_data
    )
    metrics[f"map@{k}"] = map_metric.calc(
        reco=bst_rec,
        interactions=test_data
    )

my_logger.logger.info("Метрики на тестовой выборке:", metrics)

