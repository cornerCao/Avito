import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, model_selection, metrics
from sklearn.decomposition import TruncatedSVD
from datetime import date
import os

# Training the model #
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

print(os.listdir("../input"))

train_df = pd.read_csv("../input/train.csv", parse_dates=["activation_date"])
test_df = pd.read_csv("../input/test.csv", parse_dates=["activation_date"])
print("Train file rows and columns are : ", train_df.shape)
print("Test file rows and columns are : ", test_df.shape)

train_prd = pd.read_csv("../input/periods_train.csv", parse_dates=["activation_date","date_from", "date_to"])
test_prd = pd.read_csv("../input/periods_test.csv", parse_dates=["activation_date","date_from", "date_to"])
print("Period Train file rows and columns are : ", train_prd.shape)
print("Period Test file rows and columns are : ", test_prd.shape)

# Number of days an ad was active on the portal
train_prd['days'] = (train_prd['date_to'] - train_prd['date_from']).dt.days
test_prd['days'] = (test_prd['date_to'] - test_prd['date_from']).dt.days

enc = train_prd.groupby('item_id')['days'].agg('sum').astype(np.float32).reset_index()
enc.head(5)

train_df = pd.merge(train_df, enc, how='left', on='item_id')
test_df = pd.merge(test_df, enc, how='left', on='item_id')

# New variables #
train_df["activation_weekday"] = train_df["activation_date"].dt.weekday
test_df["activation_weekday"] = test_df["activation_date"].dt.weekday

train_df["activation_month"] = train_df["activation_date"].dt.month
test_df["activation_month"] = test_df["activation_date"].dt.month

train_df["title_nwords"] = train_df["title"].apply(lambda x: len(x.split()))
test_df["title_nwords"] = test_df["title"].apply(lambda x: len(x.split()))

train_df["description"].fillna("NA", inplace=True)
test_df["description"].fillna("NA", inplace=True)
train_df["desc_nwords"] = train_df["description"].apply(lambda x: len(x.split()))
test_df["desc_nwords"] = test_df["description"].apply(lambda x: len(x.split()))

train_df['param123'] = train_df['param_1'].fillna('') + " " + train_df['param_2'].fillna('') + " " + train_df['param_3'].fillna('')
test_df['param123'] = test_df['param_1'].fillna('') + " " + test_df['param_2'].fillna('') + " " + test_df['param_3'].fillna('')

#Impute image_top_1
enc = train_df.groupby('category_name')['image_top_1'].agg(lambda x:x.value_counts().index[0]).astype(np.float32).reset_index()
enc.columns = ['category_name' ,'image_top_1_impute']
#Cross Check values
#enc = train_df.loc[train_df['category_name'] == 'Аквариум'].groupby('image_top_1').agg('count')
#enc.sort_values(['item_id'], ascending=False).head(2)

train_df = pd.merge(train_df, enc, how='left', on='category_name')
test_df = pd.merge(test_df, enc, how='left', on='category_name')

train_df['image_top_1'].fillna(train_df['image_top_1_impute'], inplace=True)
test_df['image_top_1'].fillna(test_df['image_top_1_impute'], inplace=True)

#Impute Days diff
enc = train_df.groupby('category_name')['days'].agg('median').astype(np.float32).reset_index()
enc.columns = ['category_name' ,'days_impute']
#Cross Check values
#enc = train_df.loc[train_df['category_name'] == 'Аквариум'].groupby('image_top_1').agg('count')
#enc.sort_values(['item_id'], ascending=False).head(2)

train_df = pd.merge(train_df, enc, how='left', on='category_name')
test_df = pd.merge(test_df, enc, how='left', on='category_name')

train_df['days'].fillna(train_df['days_impute'], inplace=True)
test_df['days'].fillna(test_df['days_impute'], inplace=True)


#Create image flag
test_df['image'] = test_df['image'].map(lambda x: 1 if len(str(x)) >0 else 0)
train_df['image'] = train_df['image'].map(lambda x: 1 if len(str(x)) >0 else 0)

# City names are duplicated across region, HT: Branden Murray
#https://www.kaggle.com/c/avito-demand-prediction/discussion/55630#321751
train_df['city'] = train_df['city'] + "_" + train_df['region']
test_df['city'] = test_df['city'] + "_" + test_df['region']

train_df['price'].fillna(0, inplace=True)
test_df['price'].fillna(0, inplace=True)
train_df['price'] = np.log1p(train_df['price'])
test_df['price'] = np.log1p(test_df['price'])

price_mean = train_df['price'].mean()
price_std = train_df['price'].std()
train_df['price'] = (train_df['price'] - price_mean) / price_std
test_df['price'] = (test_df['price'] - price_mean) / price_std

cat_cols = ['category_name', 'image_top_1']
num_cols = ['price', 'deal_probability']

for c in cat_cols:
    for c2 in num_cols:
        enc = train_df.groupby(c)[c2].agg(['median']).astype(np.float32).reset_index()
        enc.columns = ['_'.join([str(c), str(c2), str(c3)]) if c3 != c else c for c3 in enc.columns]
        train_df = pd.merge(train_df, enc, how='left', on=c)
        test_df = pd.merge(test_df, enc, how='left', on=c)
### TFIDF Vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,1))
#ngram_range defines how you want to have words in your dictionary.
#(min,max) = (1,2) will mean you will have unigrams and bigrms in your vocabulary.
#Example String: "The old fox"
#Vocabulary: "The", "old", "fox", "The old", "old fox"

full_tfidf = tfidf_vec.fit_transform(train_df['title'].values.tolist() + test_df['title'].values.tolist())
#train_df['title'].values.tolist() this converts all the values in the title column into a list. '+' appends two lists
train_tfidf = tfidf_vec.transform(train_df['title'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['title'].values.tolist())

### SVD Components ###
n_comp = 5
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=100000)
full_tfidf = tfidf_vec.fit_transform(train_df['description'].values.tolist() + test_df['description'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['description'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['description'].values.tolist())

### SVD Components ###
n_comp = 5
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_desc_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_desc_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=100000)
full_tfidf = tfidf_vec.fit_transform(train_df['param123'].values.tolist() + test_df['param123'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['param123'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['param123'].values.tolist())

### SVD Components ###
n_comp = 5
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_params_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_params_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
train = train_df
test = test_df

# Label encode the categorical variables #
cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1"]
for col in cat_vars:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

cols_to_drop = ["item_id", "user_id", "title", "description", "activation_date", "image", "param_2", "param_3"
                , "param123", "image_top_1_impute", "days_impute"]
train_X = train_df.drop(cols_to_drop + ["deal_probability"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

train_y = train_df["deal_probability"].values
train_id = train_df["item_id"].values
test_id = test_df["item_id"].values

# custom function to build the LightGBM model
def run_lgb(x_train, y_train, x_test):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 1000,
        "learning_rate": 0.02,
        "bagging_fraction": 0.75,
        "feature_fraction": 0.6,
        "bagging_freq": 5,
        "bagging_seed": 2018,
        "verbosity": -1,
        "max_depth": 18,
        "min_child_samples": 100
        # ,"boosting":"rf"
    }

    seed = np.random.seed(20999)
    skf = model_selection.KFold(
        n_splits=3,
        shuffle=True,
        random_state=seed)

    ntrain = len(train)
    ntest = len(test)
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((3, ntest))

    for i, (train_index, test_index) in enumerate(skf.split()):
        x_tr, x_te = x_train[train_index], x_train[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]

        lgtrain = lgb.Dataset(x_tr, label=y_tr)
        lgval = lgb.Dataset(x_te, label=y_te)
        evals_result = {}
        model = lgb.train(params, lgtrain, 2500, valid_sets=[lgval], early_stopping_rounds=50, verbose_eval=50,
                          evals_result=evals_result)

        oof_train[test_index] = model.predict(x_te)
        oof_test_skf[i, :] = model.predict(x_test, num_iteration=model.best_iteration)
        # model = lgb.cv(params, lgtrain, 1000, early_stopping_rounds=20, verbose_eval=20, stratified=False )

        print("NFOLD number:", i)
        print(model.feature_importance("split"))
        print(model.feature_importance("gain"))

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


oof_train, oof_test = run_lgb(train_X, train_y, test_X)

# Making a submission file #
oof_train = np.clip(oof_train, 0.0, 1.0)
sub_df_train = pd.DataFrame({"item_id": train_id})
sub_df_train["deal_probability"] = oof_train
sub_df_train.to_csv("baseline_lgb_train_oof.csv", index=False)

oof_test = np.clip(oof_test, 0.0, 1.0)
sub_df_test = pd.DataFrame({"item_id": test_id})
sub_df_test["deal_probability"] = oof_test
sub_df_test.to_csv("baseline_lgb_test_oof.csv", index=False)
