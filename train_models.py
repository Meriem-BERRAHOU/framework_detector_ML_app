import pandas as pd
from models.random_forest_model import train_random_forest
from models.logistic_regression_model import train_logistic_regression
from models.svm_model import train_svm
from models.fasttext_model import train_fasttext
from models.mlp_model import train_mlp
from models.xgboost_model import train_xgboost
from models.codebert_model import train_codebert_mlp

if __name__ == "__main__":

    # Entraîner différents modèles
    
    print(train_random_forest())

    print(train_logistic_regression())

    print(train_svm())

    print(train_fasttext())

    print(train_mlp())

    print(train_xgboost())

    print(train_codebert_mlp())