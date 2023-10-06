from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
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
        
        # The Encoder stays the same 
        self.encoder = ('label_encoder', ColumnTransformer(
                        transformers=[
                            ('label_encode', OrdinalEncoder(), columns_to_encode)
                            ],
                            remainder='passthrough'  # Pass columns not specified to the next step
                        ))
        
        # Define the pipelines
        if self.model == "DecisionTree":
            self.pipe = Pipeline([
                    self.encoder,
                    ('estimator', DecisionTreeClassifier(random_state=42, max_depth=24, max_leaf_nodes=1000))   
                    ],verbose=True)
            
        elif self.model == "XGBoost":
            pass
        
        elif self.model == "LightGBM":
            pass
    
    def set_estimator_params(self,**kwargs):
        """
        Sets the respective parameters of the esitmators within the pipeline.
                Args:
            kwargs (dict): Dict for params (e.g. random_state=42)
        """
        self.pipe.named_steps['estimator'].set_params(**kwargs)
        
    # Fit, predict and get the f1-score
    
    def fit(self,X,y):
        """
        This method calls the .fit()-function of the sklearn pipelinel.
        Args:
            X (pandas DataFrame): Training data (features)
            y (pandas DataFrame): Trainig data (targets)
        """
        self.pipe.fit(X,y)
        
    def predict(self,X,get_score=False):
        """
        This method calls the .predict()-function of the sklearn pipelinel.
        Args:
            X (pandas DataFrame): Data to predict (features)
        Returns:
            numpy array: predicted values
        """
        return self.pipe.predict(X)

    def get_f1_score(self,X,y):
        """
        Returns the F1-micro-score as calculated by sklearn.metrics.
        Args:
            X (pandas DataFrame): X_test
            y (pandas DataFrame): y_test
        Returns:
            float: f1-micro-score
        """
        predicted = self.pipe.predict(X)
        return f1_score(y, predicted, average='micro')
    
