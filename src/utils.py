import os, sys
import numpy as np
import pandas as pd

import dill
import pickle
from src.exception import CustomException

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, X_test, y_train, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            rs_grid = RandomizedSearchCV(model, param, cv=5)
            rs_grid.fit(X_train,y_train)

            model.set_params(**rs_grid.best_params_)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            test_model_score = accuracy_score(y_test, y_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def laod_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)