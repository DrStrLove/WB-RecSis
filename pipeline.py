#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess
import my_logger

def run_script(script_name):
    my_logger.logger.info(f"Запуск {script_name}...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    
    if result.returncode == 0:
        my_logger.logger.info(f"{script_name} выполнен успешно!")
    else:
        my_logger.logger.error(f"Ошибка в {script_name}: {result.stderr}")
        exit(1)

stages = [
    "prepare_products.py",
    "prepare_interactions_and_first_stage.py",
    "first_level_models.py",
    "prepare_second_stage.py",
    "second_stage_model.py",
    "test_model.py"
]

for stage in stages:
    run_script(stage)

my_logger.logger.info("Пайплайн успешно завершён!")


# In[ ]:




