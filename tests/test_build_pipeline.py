from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
import pandas as pd

def build_pipeline(numeric_features, categorical_features, binary_features, drop_features,target):
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

test_num_f = ['num1', 'num2']
test_cat_f = ['cat1', 'cat2']
test_bi_f = ['bi']
test_drop_f = ['drop']
test_target = ['target']
res = build_pipeline(test_num_f, test_cat_f, test_bi_f, test_drop_f, test_target)
d = {'num1': [1, 2], 'num2': [3, 4], 'cat1': ['a', 'b'], 'cat2': ['b', 'c'], 
    'bi': [0, 1], 'drop': [3, 4], 'target': [5, 6]}
df = pd.DataFrame(data=d)
X_df = df.drop(['target'], axis=1)
y_df = df['target']
res.fit(X_df, y_df)

assert res.transformers_[0][0] == 'pipeline-1'
assert type(res.transformers_[0][1].named_steps['simpleimputer']) == type(SimpleImputer())