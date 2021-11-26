# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-11-25

'''This script builds a logistic regression model for bank marketing dataset to predict if 
a banking customer will subscribe to a new product (bank term deposit) if they are contacted 
by the bank with a phone call.

Usage: logistic_regression_model.py <train_src> <test_src> <dest>

Options:
<train_src>                                     Path for the training data
<test_src>                                      Path for the testing data
<dest>                                          Path for the destination of result
'''

from get_valid_score import mean_std_cross_val_scores
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from docopt import docopt
import os, os.path
import errno

opt = docopt(__doc__)

def main(train_data, test_data, dest):
    mkdir_p(dest)
    # Split X and y
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)
    X_train, y_train = train_df.drop(columns=["y"]), train_df["y"]
    X_test, y_test = test_df.drop(columns=["y"]), test_df["y"]

    # Cross-Validation
    pipe = bulid_pipeline()
    results={}
    results["logistic regression"] = mean_std_cross_val_scores(
        pipe, X_train, y_train, cv=2, return_train_score=True
    )

    # Testing
    pipe.fit(X_train, y_train)
    accuracy = round(pipe.score(X_test, y_test), 3)
    results = results['logistic regression'].append(pd.Series(accuracy, index=["test_score"]))
    result_df = pd.DataFrame(results)
    result_df["score"] = ["fit_time", "score_time", "valid_score", "train_score", "test_score"]
    result_df = result_df.rename(columns={0:"Logistic Regression"})
    result_df = result_df[['score', 'Logistic Regression']]

    # Save the result as a csv file
    result_df.to_csv(dest+"/logistic_regression_result.csv", index=False)


def bulid_pipeline():
    categorical_features = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome"
    ]
    numeric_features = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed"
    ]
    drop_features = []
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore", sparse=False),
    )
    preprocessor =  make_column_transformer(
        ("drop", drop_features),
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features)
    )
    return make_pipeline(
        preprocessor, LogisticRegression(max_iter=1000, random_state=123, C=27.655298)
    )

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

if __name__ == "__main__":
    main(opt["<train_src>"], opt["<test_src>"], opt["<dest>"])