import os, sys
from dataclasses import dataclass

from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score ,classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting train-test input data')
            
            X_train, X_test, y_train, y_test = (
                train_arr[:,:-1],                   
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )

            models = {
                'KNN': KNeighborsClassifier(n_neighbors=1),
                'Logistic Regression': LogisticRegression(solver='lbfgs'),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'XgBoost': XGBClassifier(),
                'Lightgbm': LGBMClassifier(),
                'SVC': SVC()
            }

            model_report: dict = evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f"Model for trainig and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
        
            predicted = best_model.predict(X_test)

            score = classification_report(y_test, predicted)

            logging.info(f"Best Model for prediction")

            return score

        except Exception as e:
            raise CustomException(e, sys)