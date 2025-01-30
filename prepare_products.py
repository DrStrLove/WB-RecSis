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


my_logger.logger.info("Начало выполнения первичной обработки данных товаров.")
pq_items = pq.ParquetFile('text_data_69020_final.parquet')
my_logger.logger.info("Данные загружены.")


# In[ ]:


unique_charc_names = [
    'Коллекция',
    'Назначение платья',
    'Назначение',
    'Пол',
    'Тип карманов',
    'Тип рукава',
    'Вырез горловины',
    'Фактура материала',
    'Особенности модели',
    'Состав',
    'Длина юбки\\платья',
    'Модель платья',
    'Рисунок',
    'Вид застежки'
]


# In[ ]:


rows = []

for batch in pq_items.iter_batches():
    batch_df = batch.to_pandas()


    for index, row in batch_df.iterrows():
        characteristics = row["characteristics"]

        characteristics_list = characteristics.tolist()

        characteristics_dict = {char['charcName']: char for char in characteristics_list}

        row_data = {"nm_id": row["nm_id"]}
        
        row_data["title"] = row.get("title", 0)
        row_data["brandname"] = row.get("brandname", 0)
        
        if isinstance(row.get("colornames", None), np.ndarray):
            row_data["colornames"] = ", ".join(row["colornames"].tolist())
        else:
            row_data["colornames"] = 0 

        for char_name in unique_charc_names:
            if char_name in characteristics_dict:
                char = characteristics_dict[char_name]
                unit = char.get('unitName')
                unit_suffix = f" {unit}" if unit else ""

                if isinstance(char['charcValues'], np.ndarray):
                    values = ", ".join(char['charcValues'].tolist())
                elif char['value'] is not None:
                    values = f"{char['value']}{unit_suffix}"
                else:
                    values = "No values available."
            else:
                values = 0 
            row_data[char_name] = values

        rows.append(row_data)

final_df = pd.DataFrame(rows)


# In[ ]:


color_dict = {
    'black': ['чёрный'],
    'white': ['белый', 'бежевый', 'молочный'],
    'blue': ['синий', 'темносиний'],
    'green': ['зеленый', 'хаки'],
    'gray': ['серый', 'меланж', 'графит'],
    'pink': ['розовый'],
    'red': ['красный', 'бордовый', 'фиолетовый'],
    'light_blue': ['голубой'],
    'yellow': ['желтый'],
    'brown': ['коричневый']
}

collect_dict = {
    'spring': ['весна', 'весналето'],
    'summer': ['лето', 'весналето'],
    'fall': ['осень', 'осеньзима'],
    'winter': ['зима', 'осеньзима']
}

use_dict = {
    'comon': ['повседневная', 'повседневное', 'домашняя'],
    'office': ['офис'],
    'eve': ['вечерняя', 'вечернего'],
    'mom': ['беременных'],
    'weding': ['свадьба', 'невесты'],
    'grad': ['выпускной', 'школа'],
    'beach': ['пляж']
}

kar_dict = {
    'no': ['без', 'нет'],
    'proz': ['прорезные'],
    'nak': ['накладные']
}

ruk_dict = {
    'no': ['без'],
    'long': ['длинные'],
    'doli': ['34', '78'],
    'short': ['короткие']
}

vrez_dict = {
    'circl': ['округлый'],
    'v-shp': ['vобразный', 'v'],
    'vor': ['воротникстойка', 'воротник', 'воротничок']
}

fakt_dict = {
    'smoth': ['гладкий', 'шелковый', 'атласный'],
    'trik': ['трикотажный'],
    'text': ['текстильный'],
    'knit': ['вязаный']
}

osob_dict = {
    'breath': ['дышащий'],
    'cut': ['разрезом']
}

sost_dict = {
    'polyest': ['полиэстер'],
    'viskoz': ['вискоза'],
    'cotton': ['хлопок'],
    'elastan': ['эластан'],
    'laikr': ['лайкра'],
    'spandex': ['спандекс'],
    'len': ['лен']
}

length_dict = {
    'mini': ['мини'],
    'midi': ['миди'],
    'maxi': ['макси']
}

model_dict = {
    'no': ['без', '0'],
    'futlar': ['футляр'],
    'staight': ['прямое'],
    'trap': ['трапеция'],
    'shirt': ['рубашка'],
    'knit': ['вязаное']
}

drawing_dict = {
    'flower': ['цветочный', 'цветы'],
    'plant': ['растения'],
    'pea': ['горох'],
    'one-tone': ['однотонное', 'однотонный'],
    'shape': ['фигуры'],
    'abstract': ['абстракция'],
    'lines': ['полоска'],
    'checker': ['клетка']
}

zastk_dict = {
    'no': ['без', '0'],
    'moln': ['молния'],
    'pugov': ['пуговицы'],
    'zavaz': ['завязки', 'шнуровка']
}

def encode_categories(df):
    encoded_df = pd.DataFrame()

    df['colornames'] = df['colornames'].apply(lambda x: str(x) if pd.notna(x) else '')
    df['Коллекция'] = df['Коллекция'].apply(lambda x: str(x) if pd.notna(x) else '')
    df['Назначение платья'] = df['Назначение платья'].apply(lambda x: str(x) if pd.notna(x) else '')
    df['Тип карманов'] = df['Тип карманов'].apply(lambda x: str(x) if pd.notna(x) else '')
    df['Тип рукава'] = df['Тип рукава'].apply(lambda x: str(x) if pd.notna(x) else '')
    df['Вырез горловины'] = df['Вырез горловины'].apply(lambda x: str(x) if pd.notna(x) else '')
    df['Фактура материала'] = df['Фактура материала'].apply(lambda x: str(x) if pd.notna(x) else '')
    df['Особенности модели'] = df['Особенности модели'].apply(lambda x: str(x) if pd.notna(x) else '')
    df['Состав'] = df['Состав'].apply(lambda x: str(x) if pd.notna(x) else '')
    df['Длина юбки\\платья'] = df['Длина юбки\\платья'].apply(lambda x: str(x) if pd.notna(x) else '')
    df['Модель платья'] = df['Модель платья'].apply(lambda x: str(x) if pd.notna(x) else '')
    df['Рисунок'] = df['Рисунок'].apply(lambda x: str(x) if pd.notna(x) else '')
    df['Вид застежки'] = df['Вид застежки'].apply(lambda x: str(x) if pd.notna(x) else '')

    encoded_df['nm_id'] = df['nm_id']

    for color, names in color_dict.items():
        encoded_df[f'colr_{color}'] = df['colornames'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    for collect, names in collect_dict.items():
        encoded_df[f'collect_{collect}'] = df['Коллекция'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    for use, names in use_dict.items():
        encoded_df[f'use_{use}'] = df['Назначение платья'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    for kar, names in kar_dict.items():
        encoded_df[f'kar_{kar}'] = df['Тип карманов'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    for ruk, names in ruk_dict.items():
        encoded_df[f'ruk_{ruk}'] = df['Тип рукава'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    for vrez, names in vrez_dict.items():
        encoded_df[f'vrez_{vrez}'] = df['Вырез горловины'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    for fakt, names in fakt_dict.items():
        encoded_df[f'fakt_{fakt}'] = df['Фактура материала'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    for osob, names in osob_dict.items():
        encoded_df[f'osob_{osob}'] = df['Особенности модели'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    for sost, names in sost_dict.items():
        encoded_df[f'sost_{sost}'] = df['Состав'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    for length, names in length_dict.items():
        encoded_df[f'length_{length}'] = df['Длина юбки\\платья'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    for model, names in model_dict.items():
        encoded_df[f'model_{model}'] = df['Модель платья'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    for drawing, names in drawing_dict.items():
        encoded_df[f'drawing_{drawing}'] = df['Рисунок'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    for zastk, names in zastk_dict.items():
        encoded_df[f'zastk_{zastk}'] = df['Вид застежки'].apply(lambda x: 1 if any(name in x for name in names) else 0)
    
    return encoded_df

final_df_encoded = encode_categories(final_df)


# In[ ]:


final_df_encoded = final_df_encoded.rename(columns={'nm_id': 'item_id'})


# In[ ]:


def convert_features_to_bool(fch_df):
    columns_to_convert = [col for col in fch_df.columns if col != 'item_id']
    fch_df[columns_to_convert] = fch_df[columns_to_convert].astype(bool)
    return fch_df


# In[ ]:


final_df_encoded = convert_features_to_bool(final_df_encoded)

my_logger.logger.info("Завершение первичной обработки данных товаров.")


# In[ ]:


final_df_encoded.to_parquet("features.parquet", index=False)

my_logger.logger.info("Фичи товров сохранены.")

