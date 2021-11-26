# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-11-25

'''This script builds a random forest classifier model for bank marketing dataset to predict if 
a banking customer will subscribe to a new product (bank term deposit) if they are contacted 
by the bank with a phone call.

Usage: Build_Models.py <train_src> <test_src> <dest>

Options:
<train_src>                                     Path for the training data
<test_src>                                      Path for the testing data
<dest>                                          Path for the destination of result
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from docopt import docopt
import os, os.path
import errno

import pickle
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    average_precision_score, 
    auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    cross_val_predict
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_curve, auc

from plot_confusion_matrix import plot_confusion_mat
from get_valid_score import mean_std_cross_val_scores

import warnings
warnings.filterwarnings('ignore')

opt = docopt(__doc__)

def main(train_data, test_data, dest):
    mkdir_p(dest)
    # Split X and y
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)
    X_train, y_train = train_df.drop(columns=["y"]), train_df["y"]
    X_test, y_test = test_df.drop(columns=["y"]), test_df["y"]
    
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
    "nr.employed",
    ]

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
        "poutcome",
    ]

    binary_features = [
        "y",
    ]

    drop_features = []
    target = "y"
    
    # Cross-Validation
    results={}
    scoring_metrics = [
    "accuracy",
    "f1",
    "recall",
    "precision",
    ]
    preprocessor = build_pipeline(numeric_features,categorical_features,binary_features,drop_features,target)
    
    #CV for Dummy Classifier
    print('Creating Dummy Classifier')
    pipe_dummy = make_pipeline(preprocessor, DummyClassifier())
    results['DummyClassifier'] = mean_std_cross_val_scores(
        pipe_dummy, X_train, y_train, cv=5, return_train_score=True, scoring = scoring_metrics
    )
    
    print('Output Dummy Classifier results')
    result_df = pd.DataFrame(results).T
    result_df.to_csv(dest+"/DummyClassifier_result.csv")

    y_pred = cross_val_predict(pipe_dummy, X_train, y_train, cv=5)
    plot_confusion_mat(y_train, y_pred, 'Dummy Classifier').get_figure()
    plt.savefig(dest+"/DummyClassifier_ConMat.jpg")
    plt.clf();
    
    print('Randomized Search CV for Random Forest Classifier')
    param_grid = { 
        'RFC__max_features' : ["auto", "sqrt", "log2"],
        'RFC__min_samples_split' : range(1, 100),
        'RFC__max_depth' : range(1,5000),
        'RFC__class_weight' : ["balanced", "balanced_subsample"],
        'RFC__ccp_alpha' : 10**np.arange(-3,3, dtype=float),
    }

    pipe = Pipeline([
        ('preprocessor',preprocessor), 
        ('RFC',RandomForestClassifier(random_state=123, n_jobs=-1))
    ])

    random_search_RFC = RandomizedSearchCV(estimator=pipe,
                                           param_distributions=param_grid,
                                           n_iter = 20,
                                           n_jobs = -1,
                                           random_state = 123,
                                           return_train_score = True,
                                           scoring = scoring_metrics,
                                           refit = 'f1',
                                          )
    random_search_RFC.fit(X_train, y_train);

    print("Best hyperparameter values: ", random_search_RFC.best_params_)
    print(f"Best f1 score: {random_search_RFC.best_score_:0.3f}")

    print('Output Random Forest CV results')

    best_RFC_CV_results = pd.DataFrame(random_search_RFC.cv_results_)[[
    'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
           'params',
           'mean_train_accuracy','std_train_accuracy',
           'mean_train_f1', 'std_train_f1',
           'mean_train_recall', 'std_train_recall',
           'mean_train_precision', 'std_train_precision',
           'mean_test_accuracy','std_test_accuracy', 'rank_test_accuracy',
           'mean_test_f1','std_test_f1', 'rank_test_f1', 
           'mean_test_recall', 'std_test_recall','rank_test_recall', 
           'mean_test_precision','std_test_precision', 'rank_test_precision',
    ]].set_index("rank_test_f1").sort_index()
    best_RFC_CV_results.to_csv(dest+"/DummyClassifier_result.csv")

    print('Refit on full Train dataset')
    best_RFC_params = {key.replace('RFC__',''):val for (key, val) in random_search_RFC.best_params_.items()}
    best_RFC_params['random_state']=123
    best_RFC_params['n_jobs']=-1

    best_RFC = pipe = Pipeline([
        ('preprocessor',preprocessor), 
        ('RFC',RandomForestClassifier(**best_RFC_params))
    ])

    best_RFC.fit(X_train, y_train)

    print('Output Confusion Matrix')
    y_pred = best_RFC.predict(X_train)
    best_RFC_train_con_mat = plot_confusion_mat(y_train, y_pred,'Random Forest on Train Data').get_figure()
    best_RFC_train_con_mat.savefig(dest+"/BestRandomForest_ConMat_Train.jpg")
    plt.clf();

    y_pred = best_RFC.predict(X_test)
    best_RFC_test_con_mat = plot_confusion_mat(y_test, y_pred,'Random Forest on Test Data').get_figure()
    best_RFC_test_con_mat.savefig(dest+"/BestRandomForest_ConMat_Test.jpg")
    plt.clf();
    
    print('Output Precision and ROC curves')
    PrecisionRecallDisplay.from_estimator(best_RFC, X_test, y_test)
    plt.title("Precision Recall Curve for Random Forest")
    plt.savefig(dest+"/BestRandomForest_PrecisionCurve.jpg")
    plt.clf();
    
    y_pred = best_RFC.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (AUC ={0:.2f})'.format(roc_auc))
    plt.title("ROC for Best Random Forest model on Test data")
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")

    default_threshold = np.argmin(np.abs(thresholds - 0.5))
    plt.plot(
        fpr[default_threshold],
        tpr[default_threshold],
        "or",
        markersize=10,
        label="threshold 0.5",
    )
    plt.legend(loc="best")
    plt.savefig(dest+"/BestRandomForest_ROC.jpg")
    plt.clf();
    
    # Save model
    pickle.dump(best_RFC, open(dest+"/Best_RFC.rds", "wb"))    
    print("")


    print('Randomized Search CV for Logistics Regression')
    param_grid = { 
        'LR__C' : np.linspace(1,50,100),
        'LR__class_weight' : ["balanced", None],
    }

    pipe_lr = Pipeline([
        ('preprocessor',preprocessor), 
        ('LR',LogisticRegression(max_iter=1000, random_state=123))
    ])

    random_search_LR = RandomizedSearchCV(estimator=pipe_lr,
                                           param_distributions=param_grid,
                                           n_jobs = -1,
                                           random_state = 123,
                                           return_train_score = True,
                                           scoring = scoring_metrics,
                                           refit = 'f1',
                                          )
    random_search_LR.fit(X_train, y_train);

    # Save C vs Accuracy plot
    search_df = pd.DataFrame(random_search_LR.cv_results_).sort_values(by="param_LR__C", ascending=True)
    plt.plot(search_df["param_LR__C"], search_df["mean_test_accuracy"], label="validation")
    plt.plot(search_df["param_LR__C"], search_df["mean_train_accuracy"], label="train")
    plt.legend()
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title('Logistic Regression C vs Accuracy')
    plt.savefig(dest+'/Log_Reg_C_vs_Accuracy.png')
    plt.clf();

    print("Best hyperparameter values: ", random_search_LR.best_params_)
    print(f"Best f1 score: {random_search_LR.best_score_:0.3f}")

    print('Output Logistics Regression CV results')
    best_LR_CV_results = pd.DataFrame(random_search_LR.cv_results_)[[
    'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
           'params',
           'mean_train_accuracy','std_train_accuracy',
           'mean_train_f1', 'std_train_f1',
           'mean_train_recall', 'std_train_recall',
           'mean_train_precision', 'std_train_precision',
           'mean_test_accuracy','std_test_accuracy', 'rank_test_accuracy',
           'mean_test_f1','std_test_f1', 'rank_test_f1', 
           'mean_test_recall', 'std_test_recall','rank_test_recall', 
           'mean_test_precision','std_test_precision', 'rank_test_precision',
    ]].set_index("rank_test_f1").sort_index()
    best_LR_CV_results.to_csv(dest+"/BestLogisticsRegression_result.csv")

    print('Refit on full Train dataset')
    best_LR_params = {key.replace('LR__',''):val for (key, val) in random_search_LR.best_params_.items()}
    best_LR_params['random_state']=123
    best_LR_params['max_iter']=1000

    best_LR = Pipeline([
        ('preprocessor',preprocessor), 
        ('LR',LogisticRegression(**best_LR_params))
    ])
    
    best_LR.fit(X_train, y_train)
    
    print('Output Confusion Matrix')
    y_pred = best_LR.predict(X_train)
    best_LR_train_con_mat = plot_confusion_mat(y_train, y_pred,'Logistics Regression on Train Data').get_figure()
    best_LR_train_con_mat.savefig(dest+"/BestLogisticsRegression_ConMat_Train.jpg")
    plt.clf();
    
    y_pred = best_LR.predict(X_test)
    best_LR_test_con_mat = plot_confusion_mat(y_test, y_pred,'Logistics Regression on Test Data').get_figure()
    best_LR_test_con_mat.savefig(dest+"/BestLogisticsRegression_ConMat_Test.jpg")
    plt.clf();
    
    print('Output Precision and ROC curves')
    PrecisionRecallDisplay.from_estimator(best_LR, X_test, y_test)
    plt.title("Precision Recall Curve for Logistic Regression")
    plt.savefig(dest+"/BestLogisticsRegression_PrecisionCurve.jpg")
    plt.clf();
    
    y_pred = best_RFC.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (AUC ={0:.2f})'.format(roc_auc))
    plt.title("ROC for Best Logistic Regression model on Test data")
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")

    default_threshold = np.argmin(np.abs(thresholds - 0.5))
    plt.plot(
        fpr[default_threshold],
        tpr[default_threshold],
        "or",
        markersize=10,
        label="threshold 0.5",
    )
    plt.legend(loc="best")
    plt.savefig(dest+"/BestLogisticsRegression_ROC.jpg")
    plt.clf();
    
    # Save model
    pickle.dump(best_LR, open(dest+"/Best_LR.rds", "wb"))    
    print("")

    print('Fetch Logistics Regression Coefficients')

    categorical_features_ohe = list(
        preprocessor.named_transformers_["pipeline-2"]
        .named_steps["onehotencoder"]
        .get_feature_names_out(categorical_features)
    )
    
    new_columns = (
        numeric_features + categorical_features_ohe
    )
    
    lr_coefs = pd.DataFrame(data=best_LR[1].coef_[0], index=new_columns, columns=["Coefficient"])
    lr_coefs.to_csv(dest+"/BestLogisticsRegression_Coefficients.csv")
        

def build_pipeline(numeric_features, categorical_features, binary_features,drop_features,target):
    numeric_transformer = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler()
        )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore", sparse=False),
    )

    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features),
        ("drop", drop_features),
    )

    return preprocessor


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

if __name__ == "__main__":
    main(opt["<train_src>"], opt["<test_src>"], opt["<dest>"])





















