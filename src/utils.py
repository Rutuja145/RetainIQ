import os
import sys
import dill

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

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
                y_train, y_train_pred, zero_division=0
            )
            test_precision = precision_score(
                y_test, y_test_pred, zero_division=0
            )

            train_recall = recall_score(
                y_train, y_train_pred, zero_division=0
            )
            test_recall = recall_score(
                y_test, y_test_pred, zero_division=0
            )

            train_f1 = f1_score(
                y_train, y_train_pred, zero_division=0
            )
            test_f1 = f1_score(
                y_test, y_test_pred, zero_division=0
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
