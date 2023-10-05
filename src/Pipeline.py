# 20231005_Leo
# This class is meant to be a pipeline for the whole data-science process. 
# It is designed to be initialized and used in the main.ipynb.

class Pipeline:
    
    def __init__(self) -> None:
        pass 
        
    def import_data(self):
        pass
    
    def check_dataset(self, pandas.DataFrame df,):
        """Checks the dataset for NA values.
        Args:
            pandas (Dataframe): DataFrame to check.
        """
        if self.df.isna().any():
            print("Oh no.. you're Dataset contains NA values.")
            exit(1)
        else:
            pass
        
    def encode_features(self,type: str):
        """This function performs either a
            - OneHotEncoding ("OneHot") or
            - LabelEndoing ("Label")
        Args:
            type (str): Type of encoding.
        """
        if self.type == "OneHot":
            pass
        elif self.type == "LabeEncoder":
            pass
        else:
            exit(1)
            
        
    def transform_features(self):
        pass