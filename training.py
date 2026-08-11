import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from joblib import dump
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold,KFold,StratifiedGroupKFold,GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import (confusion_matrix,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score
                             )
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df = pd.read_csv("loan.csv")
print(df.head())
df = df.drop(columns = "Loan_ID")

X = df.drop(columns="Loan_Status")
y = df["Loan_Status"]

def check_ratio(y):
    d = {
        "count" : y.value_counts(),
        "percent" : round(y.value_counts(normalize=True), 3)
    }
    y_ratio = pd.DataFrame(d)
    return y_ratio

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

d = {
    "y_percent" : round(y.value_counts(normalize=True), 4),
    "y_train_percent" : round(y_train.value_counts(normalize=True), 4),
    "y_test_percent" : round(y_test.value_counts(normalize=True), 4)
}
y_ratio = pd.DataFrame(d)
y_ratio 

def check_uniques(df):
    for i in df.columns:
     if df[i].nunique() <= 10:
        unique_columns = df[i].unique()
        print(f"{i} --> {unique_columns}")
        print("*"*50)

check_uniques(df)

def missing_report(df):
    missing_summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum().sort_values(ascending=False),
            "missing_percent": (df.isna().mean() * 100).sort_values(ascending=False)
        }
    )
    return missing_summary
missing_report(df)

numerical = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
categorical = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Credit_History", "Dependents"]

for i in numerical:
    X_train_avg = X_train[i].mean()
    X_train[i] = X_train[i].fillna(X_train_avg)
    X_test[i] = X_test[i].fillna(X_train_avg)
missing_report(X_train)

for i in categorical:
    X_train_mode = X_train[i].mode()[0]
    X_train[i] = X_train[i].fillna(X_train_mode)
    X_test[i] = X_test[i].fillna(X_train_mode)
missing_report(X_train)

ord_bin_cat = ["Gender", "Married", "Education", "Self_Employed", "Credit_History"]
nominal_cat = ["Dependents", "Property_Area"]

ord_bin_cat = ["Gender", "Married", "Education", "Self_Employed","Credit_History"]
label_enc_classes = {}

for i in ord_bin_cat:
    le = LabelEncoder()
    le.fit(X_train[i])
    X_train[i] = le.transform(X_train[i])
    X_test[i] = le.transform(X_test[i])
    
    label_enc_classes[i] = le
    
label_enc_classes

label_enc_classes = {}
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)    
label_enc_classes = le
    
label_enc_classes

ohe = OneHotEncoder(
    drop='first',
    handle_unknown='ignore',
    sparse_output=False
)
ohe.fit(X_train[nominal_cat])
ohe_encoded_cols = ohe.get_feature_names_out()
print(ohe_encoded_cols)

train_encoded = ohe.transform(X_train[nominal_cat])
test_encoded = ohe.transform(X_test[nominal_cat])

train_df = pd.DataFrame(train_encoded, columns=ohe_encoded_cols, index=X_train.index)
test_df = pd.DataFrame(test_encoded, columns=ohe_encoded_cols, index=X_test.index)

X_train = X_train.drop(columns=nominal_cat)
X_test = X_test.drop(columns=nominal_cat)

X_train = pd.concat([X_train, train_df], axis=1)
X_test = pd.concat([X_test, test_df], axis=1)

X_train.head()

pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ("model",LogisticRegression())
])
pipeline.fit(X_train, y_train)

dump(pipeline, "Model_Dir/Loan_Model.joblib")
print("*"*50)
print("Dump Completed !")
print("*"*50)