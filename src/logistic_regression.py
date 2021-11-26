# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-11-25

'''This script builds a logistic regression model for bank marketing dataset to predict if 
a banking customer will subscribe to a new product (bank term deposit) if they are contacted 
by the bank with a phone call.
The model will return a table with the best score, a C vs Accuracy plot and a model file.

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
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from docopt import docopt
import matplotlib.pyplot as plt
import pickle
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

    tuned_pipe = bulid_pipeline(1000, 123, 1)
    search_df = log_reg_C_tunning(tuned_pipe, X_train, y_train)
    search_df = search_df.sort_values(by="rank_test_score", ascending=True)
    result_df = search_df.loc[search_df.mean_test_score == search_df.mean_test_score.max(), 'param_logisticregression__C']
    best_C = result_df.iloc[0]

    # Cross-Validation
    pipe = bulid_pipeline(1000, 123, best_C)
    results={}
    results["logistic regression"] = mean_std_cross_val_scores(
        pipe, X_train, y_train, cv=10, return_train_score=True
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

    # Save C vs Accuracy plot
    search_df = search_df.sort_values(by="param_logisticregression__C", ascending=True)
    plt.plot(search_df["param_logisticregression__C"], search_df["mean_test_score"], label="validation")
    plt.plot(search_df["param_logisticregression__C"], search_df["mean_train_score"], label="train")
    plt.legend()
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title('Logistic Regression C vs Accuracy')
    plt.savefig(dest+'/Log_Reg_C_vs_Accuracy.png')

    # Save model
    pickle.dump(pipe, open(dest+"/log_reg.rds", "wb"))

def log_reg_C_tunning(model, X_train, y_train, random_state=123):
    param_dist = {
        "logisticregression__C": loguniform(1e-3, 50)
    }
    search = RandomizedSearchCV(
        model,
        param_dist,
        verbose=1,
        n_jobs=-1,
        n_iter=50,
        return_train_score=True,
        random_state=random_state,
    )

    search.fit(X_train, y_train)
    return pd.DataFrame(search.cv_results_)[
        [
            "rank_test_score",
            "mean_test_score",
            "mean_train_score",
            "param_logisticregression__C"
        ]
    ]

def bulid_pipeline(max_iter, random_state, C):
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
        preprocessor, LogisticRegression(max_iter=max_iter, random_state=random_state, C=C)
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