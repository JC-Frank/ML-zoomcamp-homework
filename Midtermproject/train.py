#!/usr/bin/env python
# coding: utf-8

# Cleaned up version of the notebook, this is not a 1 to 1 copy. 
# Most of the prints have been removed or commented out as they can be found in the nod

import pandas as pd
# import os
import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
from sklearn.model_selection import train_test_split
# from IPython.display import display
# from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

get_ipython().run_line_magic('matplotlib', 'inline')

# data is available in the data folder, original dataset retreived from https://www.kaggle.com/datasets/joniarroba/noshowappointments/data
df = pd.read_csv("data/KaggleV2-May-2016.csv")

# # Data preparation and data cleaning
summary = pd.DataFrame({
    "count": df.count(),
    "type": df.dtypes
})

# Not all columns are named consistently so we will rename them
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns = df.columns.str.lower().str.replace('-', '_')

# No-show is a string with Yes and No so we will convert it to intergers 1 and 0 for ease of use. 
df["no_show_int"] = (df["no_show"] == "Yes").astype(int)

# We have the dates of the appointment and when it was scheduled in appointmentday and scheduledday respectively. 
# From those dates we can calculate how longago an appointment was made, which might have an influence on the no_show value. 
df["appointmentday"] = pd.to_datetime(df["appointmentday"])
df["scheduledday"] = pd.to_datetime(df["scheduledday"])

# as scheduledday does not have hours we make sure we only use the date and not the hours. 
df["wait_time"] = (df["appointmentday"].dt.normalize() - df["scheduledday"].dt.normalize()).dt.days

# there are 4 rows where the scheduledday was 1 day after the appointment day and 1 row where it was 6 days before. 
df = df[df["wait_time"] >= 0]

# Age has at least 1 negative number which will have to be removed. 
df = df[df["age"] >= 0]

# 5 subjects with an age of 115 and none between 102 and 115 seems off, i will be removing those rows as they do not seem reliable. 
df = df[df["age"] < 115]

# create age_group variable
df["age_group"] = pd.cut(
    df["age"],
    bins=list(range(0, 71, 5)) + [df["age"].max() + 1],
    right=False
).astype(str)



# Lets set up some standard lists so we can use them later
categorical = [
    'gender',
    'age_group', 
    'sms_received',
    'hipertension',
    'diabetes',
    'alcoholism',
    'handcap',
    'scholarship',
    'neighbourhood'
]

numerical = [
    'age',
    'wait_time'
]

# ### split the datasets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# reset the index
df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# isolate the target
y_full_train = df_full_train.no_show_int.values
y_train = df_train.no_show_int.values
y_val = df_val.no_show_int.values
y_test = df_test.no_show_int.values

# delete the no_show columns
del df_full_train['no_show_int']
del df_train['no_show_int']
del df_val['no_show_int']
del df_test['no_show_int']

del df_full_train['no_show']
del df_train['no_show']
del df_val['no_show']
del df_test['no_show']

# training and validation
train_dicts = df_train[categorical + numerical].to_dict(orient="records")
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dicts)


# Tune max_depth
scores = []

for max_depth in [5, 10, 15, 20, 25]:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=max_depth,
        random_state=1,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    print(
        f"max_depth={max_depth:2d} | "
        f"AUC={auc:.3f}"
    )

max_depth = 20

# Tune min_samples_leaf
scores = []

for min_samples_leaf in [1, 2, 3, 4, 5, 10, 15, 20, 50, 100]:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=1,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    print(
        f"min_samples_leaf={min_samples_leaf} | "
        f"AUC={auc:.3f}"
    )

min_samples_leaf = 3 # output is about equal for 1 to 5, notebook showed 3 to be slightly better overall

# Train the final Model
dicts_full_train = df_full_train[categorical + numerical].to_dict(orient="records")
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test[categorical + numerical].to_dict(orient="records")
X_test = dv.transform(dicts_test)

# Final Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    random_state=1
)

rf.fit(X_full_train, y_full_train)

y_pred = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)


print(f'final auc={auc}')

# Save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

print(f'the model is saved to {output_file}')


