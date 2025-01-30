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


X_train = pd.read_parquet("X_train.parquet")
y_train = pd.read_parquet("y_train.parquet").squeeze()

X_es = pd.read_parquet("X_es.parquet")
y_es = pd.read_parquet("y_es.parquet").squeeze()

X_val = pd.read_parquet("X_val.parquet")
y_val = pd.read_parquet("y_val.parquet").squeeze()


# In[ ]:


my_logger.logger.info("Начало обучения CatBoost.")

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.1,
    depth=6,
    eval_metric='AUC',
    use_best_model=True,
    early_stopping_rounds=50,
    random_state=42,
    verbose=10
)

model.fit(
    X_train, y_train,
    eval_set=(X_es, y_es)
)

my_logger.logger.info("Обучение Catboost успешно.")

catboost_model_path = "catboost_model.cbm"
joblib.dump(model, catboost_model_path)
my_logger.logger.info(f"CatBoost model saved to {catboost_model_path}")

y_val_pred_proba = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred_proba)
my_logger.logger.info("Validation ROC AUC:", roc_auc)


# In[ ]:


train_pool = Pool(data=X_train, label=y_train)

feature_importances = model.get_feature_importance(train_pool)

feature_names = X_train.columns
fi_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)


# In[ ]:


fi_df.to_parquet("fi_df.parquet", index=False)

