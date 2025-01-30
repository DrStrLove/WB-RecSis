#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import my_logger
import os
import re
import json
import joblib
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


my_logger.logger.info("Начало выполнения первичной обработки взаимодействий.")
dt_user = pd.read_parquet("train_data_10_10_24_10_11_24_final.parquet")
my_logger.logger.info("Данные загружены.")


# In[ ]:


dt_user = dt_user.drop(columns=['date'])
dt_user = dt_user.drop(columns=['subject_id'])

user_interactions = dt_user['wbuser_id'].value_counts()
item_interactions = dt_user['nm_id'].value_counts()

valid_users = user_interactions[(user_interactions >= 5) & (user_interactions <= 15)].index
valid_items = item_interactions[item_interactions >= 2].index

filtered_data = dt_user[dt_user['wbuser_id'].isin(valid_users) & dt_user['nm_id'].isin(valid_items)]

filtered_data = filtered_data.drop_duplicates(subset=['wbuser_id', 'nm_id'])

my_logger.logger.info("Завершение первичной обработки взаимодействий.")


# In[ ]:


my_logger.logger.info("Начало подготовки выборок первого этапа и теста.")

day_1_data = filtered_data[filtered_data['dt'].dt.date == pd.Timestamp('2024-10-10').date()]
day_2_data = filtered_data[filtered_data['dt'].dt.date == pd.Timestamp('2024-10-11').date()]

cold_users = set(day_2_data['wbuser_id']) - set(day_1_data['wbuser_id'])
cold_users_data = day_2_data[day_2_data['wbuser_id'].isin(cold_users)]

cold_users_sample = cold_users_data['wbuser_id'].drop_duplicates()
cold_users_sample = cold_users_sample.sample(frac=0.1, random_state=42)

cold_users_data = cold_users_data[cold_users_data['wbuser_id'].isin(cold_users_sample)]

first_val_cold_users, cold_test_users = train_test_split(cold_users_sample, test_size=0.5, random_state=42)

first_val_cold = cold_users_data[cold_users_data['wbuser_id'].isin(first_val_cold_users)]
cold_test = cold_users_data[cold_users_data['wbuser_id'].isin(cold_test_users)]

warm_users_data = day_2_data[~day_2_data['wbuser_id'].isin(cold_users)]

warm_users_over_2_interactions = warm_users_data.groupby('wbuser_id').filter(lambda x: len(x) > 2)

warm_users_2_or_less = warm_users_data.groupby('wbuser_id').filter(lambda x: len(x) <= 2)

valid_warm_users = []
used_first_day_indices = []
for user_id, interactions in warm_users_2_or_less.groupby('wbuser_id'):
    first_day_interactions = day_1_data[day_1_data['wbuser_id'] == user_id]
    if len(first_day_interactions) > 4:
        last_2_first_day = first_day_interactions.sort_values(by='dt', ascending=False).head(2)
        used_first_day_indices.extend(last_2_first_day.index)
        user_warm_data = pd.concat([last_2_first_day, interactions], ignore_index=True)
        valid_warm_users.append(user_warm_data)

remaining_day_1_data = day_1_data.drop(index=used_first_day_indices)

warm_users_combined = pd.concat([warm_users_over_2_interactions] + valid_warm_users, ignore_index=True)

last_interactions = warm_users_combined.sort_values(by='dt', ascending=False).groupby('wbuser_id').head(4)

remaining_warm_users_data = warm_users_combined.drop(last_interactions.index)

warm_users = last_interactions['wbuser_id'].drop_duplicates()
warm_val_users, warm_test_users = train_test_split(warm_users, test_size=0.5, random_state=42)

first_val_warm = last_interactions[last_interactions['wbuser_id'].isin(warm_val_users)]
warm_test = last_interactions[last_interactions['wbuser_id'].isin(warm_test_users)]

train = pd.concat([remaining_day_1_data, remaining_warm_users_data], ignore_index=True)


# In[ ]:


first_val_warm = first_val_warm.rename(columns={'wbuser_id': 'user_id', 'nm_id': 'item_id', 'dt': 'datetime'})
first_val_cold = first_val_cold.rename(columns={'wbuser_id': 'user_id', 'nm_id': 'item_id', 'dt': 'datetime'})
warm_test = warm_test.rename(columns={'wbuser_id': 'user_id', 'nm_id': 'item_id', 'dt': 'datetime'})
cold_test = cold_test.rename(columns={'wbuser_id': 'user_id', 'nm_id': 'item_id', 'dt': 'datetime'})
train = train.rename(columns={'wbuser_id': 'user_id', 'nm_id': 'item_id', 'dt': 'datetime'})


# In[ ]:


train.insert(2, 'weight', 1)
warm_test.insert(2, 'weight', 1)
cold_test.insert(2, 'weight', 1)
first_val_cold.insert(2, 'weight', 1)
first_val_warm.insert(2, 'weight', 1)

my_logger.logger.info("Завершение подготовки выборок первого этапа и теста.")


# In[ ]:


first_val_warm.to_parquet("first_val_warm.parquet", index=False)
first_val_cold.to_parquet("first_val_cold.parquet", index=False)
warm_test.to_parquet("warm_test.parquet", index=False)
cold_test.to_parquet("cold_test.parquet", index=False)
train.to_parquet("train.parquet", index=False)

my_logger.logger.info("Выборки первого этапа и теста сохранены.")

