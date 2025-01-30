#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import my_logger
import os
import re
import json
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


my_logger.logger.info("Начало работы моделей первого уровня.")
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# In[ ]:


file_val_warm = "first_val_warm.parquet"
file_train = "train.parquet"

df_val_w1 = pd.read_parquet(file_val_warm)
df_train = pd.read_parquet(file_train)

my_logger.logger.info("Данные загружены.")


# In[ ]:


dataset_train = Dataset.construct(df_train)
dataset_val_w1 = Dataset.construct(df_val_w1)

val_users = df_val_w1[Columns.User].unique()


# In[ ]:


ks = [10, 30, 50, 100]


# In[ ]:


def calculate_metrics(model, dataset_train, interactions_df, val_users, ks):
    recommendations = model.recommend(
        users=val_users,
        dataset=dataset_train,
        k=max(ks),
        filter_viewed=True
    )

    metrics = {}
    for k in ks:
        recall_metric = Recall(k=k)
        hitrate_metric = HitRate(k=k)
        precision_metric = Precision(k=k)
        map_metric = MAP(k=k)

        metrics[f"recall@{k}"] = recall_metric.calc(
            reco=recommendations,
            interactions=interactions_df
        )
        metrics[f"hitrate@{k}"] = hitrate_metric.calc(
            reco=recommendations,
            interactions=interactions_df
        )
        metrics[f"precision@{k}"] = precision_metric.calc(
            reco=recommendations,
            interactions=interactions_df
        )
        metrics[f"map@{k}"] = map_metric.calc(
            reco=recommendations,
            interactions=interactions_df
        )
    return metrics


# In[ ]:


my_logger.logger.info("Начало работы Popular.")
model = PopularModel()
model.fit(dataset_train)

popular_model_filename = "popular_model.pkl"
model.save(popular_model_filename)

my_logger.logger.info(f"Модель популярное сохранена в {popular_model_filename}")


# In[ ]:


all_metrics = []

for k in ks:
    metrics = calculate_metrics(model, dataset_train, df_val_w1, val_users, [k])
    metrics.update({"k": k})
    all_metrics.append(metrics)

for metrics in all_metrics:
    my_logger.logger.info(f"Метрики для k={metrics['k']}: {metrics}")


# In[ ]:


recommendations = model.recommend(
    users=val_users,
    dataset=dataset_train,
    k=50,
    filter_viewed=True
)
recommendations.to_parquet("rec_pop.parquet", index=False)

my_logger.logger.info("Завершение работы Popular. Рекомендации сохранены")


# In[ ]:


my_logger.logger.info("Начало работы KNN.")

k_list = [20]
models = {
    "cos": ItemItemRecommender
}


# In[ ]:


results = []

for model_name, model_class in models.items():
    for k in k_list:
        my_logger.logger.info(f"Training {model_name} model with k={k}")

        recommender_model = model_class(K=k)
        model = ImplicitItemKNNWrapperModel(recommender_model)

        model.fit(dataset_train)

        model_filename = "knn_model.pkl"
        model.save(model_filename)
        my_logger.logger.info(f"KNN model '{model_name}' with k={k} saved to {model_filename}")

        metrics = calculate_metrics(model, dataset_train, df_val_w1, val_users, ks)
        metrics.update({
            "k": k,
            "model": model_name
        })
        results.append(metrics)

        recommendations = model.recommend(
            users=val_users,
            dataset=dataset_train,
            k=50,
            filter_viewed=True
        )

        if 50 in ks:
            output_filename = f"rec_knn_{model_name}.parquet"
            recommendations.to_parquet(output_filename, index=False)

        my_logger.logger.info(f"Metrics for {model_name} model with k={k}: {metrics}")
my_logger.logger.info("Завершение работы KNN. Рекомендации сохранены")


# In[ ]:


my_logger.logger.info("Начало работы iASL.")

factors_list = [256]
regularization_list = [0.5]
iterations_list = [6]
alpha_list = [500]


# In[ ]:


results = []
for factors in factors_list:
    for reg in regularization_list:
        for iterations in iterations_list:
            for alpha in alpha_list:
                my_logger.logger.info(f"Training model with factors={factors}, reg={reg}, iterations={iterations}, alpha={alpha}")

                als_model = AlternatingLeastSquares(
                    factors=factors,
                    regularization=reg,
                    iterations=iterations,
                    alpha=alpha,
                    use_gpu=False
                )

                model = ImplicitALSWrapperModel(als_model)
                model.fit(dataset_train)

                als_model_filename = "als_model.pkl"
                model.save(als_model_filename)
                my_logger.logger.info(f"ALS model saved to {als_model_filename}") 
                
                metrics = calculate_metrics(model, dataset_train, df_val_w1, val_users, ks)
                metrics.update({
                    "factors": factors,
                    "regularization": reg,
                    "iterations": iterations,
                    "alpha": alpha
                })
                results.append(metrics)

                my_logger.logger.info(f"Metrics for factors={factors}, reg={reg}, iterations={iterations}, alpha={alpha}: {metrics}")

                recommendations = model.recommend(
                    users=val_users,
                    dataset=dataset_train,
                    k=50,
                    filter_viewed=True
                )
                recommendations.to_parquet(
                    f"rec_als_{factors}_{reg}_{iterations}_{alpha}.parquet",
                    index=False
                )
my_logger.logger.info("Завершение работы iASL. Рекомендации сохранены")

