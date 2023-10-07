from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# from category_encoders import
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import xgboost as xgb
import pandas as pd
import numpy as np

# 20231005_Leo
# This class is meant to calls several sklearn-pipelines.
# It is designed to be initialized and used in the main.ipynb.


class MyModel:
    def __init__(self, model, columns_to_labelencode, columns_to_targetencode):
        """
        Constructor for Sklean-Pipelines.
        The following pipelines ares available:
            - DecisionTree
            - XGBoost
            - LightGBM
        Args:
            type (str): Types
        """

        self.model = model
        self.columns_to_labelencode = columns_to_labelencode
        self.columns_to_targetencode = columns_to_targetencode

        # The Encoder stays the same
        self.encoder = (
            "label_encoder",
            ColumnTransformer(
                transformers=[
                    ("label_encode", OrdinalEncoder(), columns_to_labelencode),
                    (
                        "target",
                        TargetEncoder(random_state=0, target_type="continuous"),
                        columns_to_targetencode,
                    ),
                ],
                remainder="passthrough",
            ),
        )

        # Define the pipelines
        if self.model == "DecisionTree":
            self.pipe = Pipeline(
                [
                    self.encoder,
                    (
                        "estimator",
                        DecisionTreeClassifier(
                            random_state=42, max_depth=24, max_leaf_nodes=1000
                        ),
                    ),
                ],
                verbose=True,
            )

        elif self.model == "XGBoost":
            self.pipe = Pipeline(
                [
                    self.encoder,
                    (
                        "estimator",
                        xgb.XGBClassifier(
                            n_estimators=1000,
                            learning_rate=0.1,
                            max_depth=5,
                            min_child_weight=1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            gamma=0,
                            reg_lambda=1e-5,
                            alpha=1e-5,
                            objective="binary:logistic",  # for binary classification
                            eval_metric="logloss",  # use logloss as the evaluation metric
                            random_state=42,  # set a random seed for reproducibility
                        ),
                    ),  # Learning rate = 0.05, 1000 rounds, max depth = 3-5, subsample = 0.8-1.0, colsample_bytree = 0.3 - 0.8, lambda = 0 to 5
                ],
                verbose=True,
            )

        elif self.model == "LightGBM":
            pass

    def set_estimator_params(self, **kwargs):
        """
        Sets the respective parameters of the esitmators within the pipeline.
                Args:
            kwargs (dict): Dict for params (e.g. random_state=42)
        """
        self.pipe.named_steps["estimator"].set_params(**kwargs)

    # Fit, predict and get the f1-score
    def fit(self, X, y):
        """
        This method calls the .fit()-function of the sklearn pipelinel.
        Args:
            X (pandas DataFrame): Training data (features)
            y (pandas DataFrame): Trainig data (targets)
        """
        # XGBoost needs the target-variable from 0-2, not 1-3: https://stackoverflow.com/questions/71996617/invalid-classes-inferred-from-unique-values-of-y-expected-0-1-2-3-4-5-got
        if self.model == "XGBoost":
            y = y - 1

        self.pipe.fit(X, y)

    def predict(self, X):
        """
        This method calls the .predict()-function of the sklearn pipelinel.
        Args:
            X (pandas DataFrame): Data to predict (features)
        Returns:
            numpy array: predicted values
        """
        y_predicted = self.pipe.predict(X)
        # XGBoost needs the target-variable from 0-2, not 1-3: https://stackoverflow.com/questions/71996617/invalid-classes-inferred-from-unique-values-of-y-expected-0-1-2-3-4-5-got
        if self.model == "XGBoost":
            y_predicted = y_predicted + 1

        return y_predicted

    def predict2submit(self, X, X_ids):
        """
        This function conducts the prediction and formats the output for submitting.
        Args:
            X (pandas DataFrame): Data to predict and to submit
        """
        y_predicted = self.predict(X)
        submit = pd.concat(
            [X_ids.loc[:, "building_id"], pd.Series(y_predicted)], axis=1
        )
        submit.columns = ["building_id", "damage_grade"]

        return submit

    def get_f1_score(self, X, y):
        """
        Returns the F1-micro-score as calculated by sklearn.metrics.
        Args:
            X (pandas DataFrame): X_test
            y (pandas DataFrame): y_test
        Returns:
            float: f1-micro-score
        """
        y_predicted = self.predict(X)

        return f1_score(y, y_predicted, average="micro")

    def get_feature_importance(self, X_columns):
        """
        Returns a dict of sorted feature importance according to the used estimator model.
        Args:
            X_columns (pandas DataFrame): X_train.columns

        Returns:
            dict: feature_importance
        """
        feature_importances = self.pipe.named_steps["estimator"].feature_importances_
        features = dict(zip(X_columns, feature_importances))

        return dict(sorted(features.items(), key=lambda item: item[1], reverse=True))
