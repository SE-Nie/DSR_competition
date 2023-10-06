from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from typing import Literal

# 20231005_Leo
# This class is meant to calls several sklearn-pipelines. 
# It is designed to be initialized and used in the main.ipynb.

class Models:
    
    def __init__(self, model,columns_to_encode):
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
        self.columns_to_encode = columns_to_encode
        
        if self.model == "DecisionTree":
            self.pipe = Pipeline([
                    ('label_encoder', ColumnTransformer(
                        transformers=[
                            ('label_encode', OrdinalEncoder(), self.columns_to_encode)
                            ],
                            remainder='passthrough'  # Pass columns not specified to the next step
                        )),
    
                        ('estimator', DecisionTreeClassifier(random_state=42, max_depth=24, max_leaf_nodes=1000))   
                    ],verbose=True)
            
        elif self.model == "XGBoost":
            pass
        
        elif self.model == "LightGBM":
            pass
        
    def fit(self,X,y):
        """
        This method calls the .fit()-function of the sklearn pipelinel.
        Args:
            X (pandas DataFrame): Training data (features)
            y (pandas DataFrame): Trainig data (targets)
        """
        self.pipe.fit(X,y)
        
    def predict(self,X):
        """
        This method calls the .predict()-function of the sklearn pipelinel.
        Args:
            X (pandas DataFrame): Data to predict (features)
        """
        return self.pipe.predict(X)

    def define_estimator_params(self,**kwargs):
        """_summary_
        """
        for i in kwargs:
            s = f"estimator__{i}={kwargs[i]}"
            print(s)
            self.pipe.set_params(s)
        
    
