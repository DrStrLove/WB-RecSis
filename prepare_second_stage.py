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
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import (
    ItemItemRecommender, TFIDFRecommender, BM25Recommender
)


# In[ ]:


my_logger.logger.info("Начало подготовки данных второго этапа.")

file_val = 'first_val_warm.parquet'

file_rec_1 = 'rec_als_256_0.5_6_500.parquet'
file_rec_2 = 'rec_knn_cos.parquet'
file_rec_4 = 'rec_pop.parquet'


# In[ ]:


df_val_w1 = pd.read_parquet(file_val)

rec_als = pd.read_parquet(file_rec_1)
rec_cos = pd.read_parquet(file_rec_2)
rec_pop = pd.read_parquet(file_rec_4)

rec_als = rec_als.rename(columns={"score": "als_score", "rank": "als_rank"})
rec_cos = rec_cos.rename(columns={"score": "cos_score", "rank": "cos_rank"})
rec_pop = rec_pop.rename(columns={"score": "pop_score", "rank": "pop_rank"})

bst_rec = rec_pop
bst_rec = bst_rec.merge(rec_als, on=["user_id", "item_id"], how="outer")
bst_rec = bst_rec.merge(rec_cos, on=["user_id", "item_id"], how="outer")


# In[ ]:


bst_rec = bst_rec.fillna(0)

my_logger.logger.info("Данные загружены и объединены.")


# In[ ]:


my_logger.logger.info("Добавление таргета.")
df_target = bst_rec[['user_id', 'item_id']].merge(
    df_val_w1[['user_id', 'item_id']],
    on=['user_id', 'item_id'],
    how='left',
    indicator=True
)

bst_rec['target'] = (df_target['_merge'] == 'both').astype(int)

del df_target


# In[ ]:


invalid_user_ids = bst_rec.groupby('user_id')['target'].sum()
invalid_user_ids = invalid_user_ids[invalid_user_ids == 0].index

bst_rec = bst_rec[~bst_rec['user_id'].isin(invalid_user_ids)]


# In[ ]:


target_1_count = bst_rec[bst_rec['target'] == 1].shape[0]
target_0_count = bst_rec[bst_rec['target'] == 0].shape[0]

my_logger.logger.info(f"Число строк с target = 1: {target_1_count}")
my_logger.logger.info(f"Число строк с target = 0: {target_0_count}")


# In[ ]:


negative_indices = bst_rec[bst_rec['target'] == 0].sample(frac=0.04, random_state=42).index

bst_rec = bst_rec[(bst_rec['target'] == 1) | (bst_rec.index.isin(negative_indices))]

bst_rec.reset_index(drop=True, inplace=True)

my_logger.logger.info("Число строк после изменений:", len(bst_rec))


# In[ ]:


target_1_count = bst_rec[bst_rec['target'] == 1].shape[0]
target_0_count = bst_rec[bst_rec['target'] == 0].shape[0]

my_logger.logger.info(f"Число строк с target = 1 после изменения: {target_1_count}")
my_logger.logger.info(f"Число строк с target = 0 после изменения: {target_0_count}")

my_logger.logger.info("Классы сбалансированы.")


# In[ ]:


my_logger.logger.info("Добавление фичей.")

train_file = "train.parquet"
file_fch = "features.parquet"

history = pd.read_parquet(train_file)
fch_df = pd.read_parquet(file_fch)


# In[ ]:


unique_items = history['item_id'].unique()
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
    calculate_mean_normalized_cooccurrence(row['user_id'], row['item_id'])
    for _, row in bst_rec.iterrows()
]


# In[ ]:


item_popularity = history.groupby('item_id')['user_id'].nunique().reset_index()
item_popularity.rename(columns={'user_id': 'it_pop'}, inplace=True)

bst_rec = bst_rec.merge(item_popularity, on='item_id', how='left')

bst_rec['it_pop'] = bst_rec['it_pop'].fillna(0).astype(int)


# In[ ]:


bst_rec = bst_rec.merge(
    fch_df,
    on='item_id',
    how='left'
)
bst_rec.fillna(0, inplace=True)

my_logger.logger.info("Добавление фичей успешно.")


# In[ ]:


my_logger.logger.info("Подготовка выборок второго этапа.")

features_to_drop = ['user_id', 'item_id', 'target']
X = bst_rec.drop(columns=features_to_drop) 
y = bst_rec['target']                      

X_temp, X_val, y_temp, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_es, y_train, y_es = train_test_split(
    X_temp,
    y_temp,
    test_size=0.1,
    random_state=42,
    stratify=y_temp
)

my_logger.logger.info(f"Train size: {X_train.shape[0]}")
my_logger.logger.info(f"Early stopping size: {X_es.shape[0]}")
my_logger.logger.info(f"Validation size: {X_val.shape[0]}")

my_logger.logger.info("Подготовка выборок второго этапа завершена.")


# In[ ]:


X_train.to_parquet("X_train.parquet", index=False)
y_train.to_frame().to_parquet("y_train.parquet", index=False)

X_es.to_parquet("X_es.parquet", index=False)
y_es.to_frame().to_parquet("y_es.parquet", index=False)

X_val.to_parquet("X_val.parquet", index=False)
y_val.to_frame().to_parquet("y_val.parquet", index=False)

