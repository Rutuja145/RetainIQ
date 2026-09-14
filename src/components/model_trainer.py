import os
import sys

from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object, evaluate_model

print("MODEL TRAINER FILE LOADED")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000
                ),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
            }

            param_grid = {
                "C": [0.01, 0.1, 1, 10, 100]
            }

            grid_search = GridSearchCV(
                estimator=models["Logistic Regression"],
                param_grid=param_grid,
                scoring="f1",
                cv=5,
                n_jobs=-1
            )

            grid_search.fit(x_train, y_train)

            print("Best Parameters:", grid_search.best_params_)
            print("Best CV F1:", grid_search.best_score_)

            models["Logistic Regression"] = grid_search.best_estimator_

            model_report: dict = evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models
            )

            print("\nModel Comparison:")
            for model_name, metrics in model_report.items():
                print(model_name, metrics)


            best_model_name = "Logistic Regression"

            best_model_score = grid_search.best_score_

            best_model = models[best_model_name]

            

            logging.info("best found model on both training and testing dataset")   

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted, zero_division=0)
            recall = recall_score(y_test, predicted, zero_division=0)
            f1 = f1_score(y_test, predicted, zero_division=0)

            logging.info(f"Best Model Accuracy: {accuracy}")
            logging.info(f"Best Model Precision: {precision}")
            logging.info(f"Best Model Recall: {recall}")
            logging.info(f"Best Model F1 Score: {f1}")

            return {
                "model": best_model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        except Exception as e:
            raise CustomException(e, sys)

            

        
    def evaluate_model(x_train, y_train, x_test, y_test, models):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]

                model.fit(x_train, y_train)

                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)

                train_precision = precision_score(
                    y_train,
                    y_train_pred,
                    zero_division=0
                )
                test_precision = precision_score(
                    y_test,
                    y_test_pred,
                    zero_division=0
                )

                train_recall = recall_score(
                    y_train,
                    y_train_pred,
                    zero_division=0
                )
                test_recall = recall_score(
                    y_test,
                    y_test_pred,
                    zero_division=0
                )

                train_f1 = f1_score(
                    y_train,
                    y_train_pred,
                    zero_division=0
                )
                test_f1 = f1_score(
                    y_test,
                    y_test_pred,
                    zero_division=0
                )

                test_roc_auc = roc_auc_score(
                    y_test,
                    model.predict_proba(x_test)[:, 1]
                )

                report[list(models.keys())[i]] = {
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "train_precision": train_precision,
                    "test_precision": test_precision,
                    "train_recall": train_recall,
                    "test_recall": test_recall,
                    "train_f1": train_f1,
                    "test_f1": test_f1,
                    "test_roc_auc": test_roc_auc
                }

            return report

        except Exception as e:
            raise CustomException(e, sys)

