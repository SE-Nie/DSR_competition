from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
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
    
                        ('estimator', DecisionTreeClassifier(random_state=42, max_depth=4, max_leaf_nodes=1000))   
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
        """_summary_

        Args:
            X (_type_): _description_
        """

        
        
    
