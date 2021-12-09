from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer

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
test_drop_f = ['drop']
test_target = ['target']
res = build_pipiline(test_num_f, test_cat_f, test_drop_f, test_target)
print(res)
print('--------')