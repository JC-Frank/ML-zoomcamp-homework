#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression




get_ipython().run_line_magic('matplotlib', 'inline')


# # Document data source and retrieval steps.

# In[ ]:


# data is available in the data folder, original dataset retreived from https://www.kaggle.com/datasets/joniarroba/noshowappointments/data
df = pd.read_csv("data/KaggleV2-May-2016.csv")


# # Data preparation and data cleaning

# In[ ]:


print("head")
print(df.head())

summary = pd.DataFrame({
    "count": df.count(),
    "type": df.dtypes
})

print(summary)


# Not all columns are named consistently so we will rename them

# In[ ]:


df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns = df.columns.str.lower().str.replace('-', '_')


# No-show is a string with Yes and No so we will convert it to intergers 1 and 0 for ease of use. 
# 

# In[ ]:


df["no_show_int"] = (df["no_show"] == "Yes").astype(int)


# We have the dates of the appointment and when it was scheduled in appointmentday and scheduledday respectively. From those dates we can calculate how longago an appointment was made, which might have an influence on the no_show value. 

# In[ ]:


df["appointmentday"] = pd.to_datetime(df["appointmentday"])
df["scheduledday"] = pd.to_datetime(df["scheduledday"])

# as scheduledday does not have hours we make sure we only use the date and not the hours. 
df["wait_time"] = (df["appointmentday"].dt.normalize() - df["scheduledday"].dt.normalize()).dt.days
print("Minimum:", df["wait_time"].min())
print("Maximum:", df["wait_time"].max())


# There are days where the scheduledday are before the appointment days. Lets see how often this happens and remove those rows as they cannot be valid

# In[ ]:


df["wait_time"].value_counts().sort_index()


# there are 4 rows where the scheduledday was 1 day after the appointment day and 1 row where it was 6 days before. 

# In[ ]:


df = df[df["wait_time"] >= 0]


# Lets set up some standard lists so we can use them later

# In[ ]:


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


# # Explore distributions, missing values, correlations, outliers.

# In[ ]:


## missing values:
df.isnull().sum()


# Check the descriptive statistics of the dataset

# In[ ]:


df.describe()


# Age has at least 1 negative number which will have to be removed. Age also has a maximum of 115 which seems a bit high, lets take a look if there is an outlier which we are better of removing.
# 
# 
# handcap has a min value of 0 and a max of 4 which might indicate severity and is not just 0 and 1 for a yes/no. however the quartiles give an indication that there are either very little values of 1 to 4 or that there is an error in the data. 

# In[ ]:


df = df[df["age"] >= 0]
df.loc[df["age"] > 80, "age"].value_counts().sort_index()


# 5 subjects with an age of 115 and none between 102 and 115 seems off, i will be removing those rows as they do not seem reliable. 

# In[ ]:


df = df[df["age"] < 115]


# Handcap had quantiles of all zeros and a maximum of 4, lets quickly check if there is an issue with the data or if the counts are just very low. 

# In[ ]:


df["handcap"].value_counts().sort_index()


# The counts for handcap seem to just be very low. 

# ## Distributions
# main the feature that we will predict is the 'No-show' column
# We will thus first look at the distribution of this feature itself and how it relates to other features. 

# ### No-show

# In[ ]:


sns.countplot(df, x = "no_show")
plt.title("No Show distribution")


# In[ ]:


# Percentage of No-show:
df["no_show"].value_counts(normalize = True).round(4)*100


# ### Age

# In[ ]:


# Distribution of Age
age_hist = sns.histplot(df.age)
age_hist.plot(
    kind="hist",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution")
plt.show()


# In[ ]:


age_hist = sns.histplot(df.age, bins=range(0, int(df["age"].max()) + 6, 5))
age_hist.plot(
    kind="hist",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution, bin every 5 years")
plt.show()


# As there is a clear dropoff after 65+ which might make the prediction on higher age less reliable. I will make an age_group variable cutting off after 70 to maybe use this later to increase the reliablitlity for older subjects. 
# 

# In[ ]:


df["age_group"] = pd.cut(
    df["age"],
    bins=list(range(0, 71, 5)) + [df["age"].max() + 1],
    right=False
).astype(str)

# check the percentages if these groups are relatively equal
df["age_group"].value_counts(normalize = True).round(4)*100


# In[ ]:


age_counts = pd.crosstab(
    df["age_group"],
    df["no_show_int"]
)

age_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("Age group")
plt.ylabel("Count")
plt.title("Attendance vs No-show by age group")
plt.legend(["Attended", "No-show"])
plt.xticks(rotation=0)
plt.show()


# In[ ]:


age_rate = pd.crosstab(
    df["age_group"],
    df["no_show_int"],
    normalize="index"
) * 100

age_rate.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("Age group")
plt.ylabel("Percentage (%)")
plt.title("Attendance vs No-show by age group")
plt.legend(["Attended", "No-show"])
plt.xticks(rotation=0)
plt.show()


# In[ ]:


df.groupby("age_group")["no_show_int"].mean().mul(100)


# Age groups are relatively equally represented in the attendance records. however it does seem like we could interpret a slight increase in attendance the older a person is. With the exception of 0 to 10, which could be caused by them being dependant on their parents/guardians. 
# 

# ### Gender

# Check how Gender is distributed including compared to age and no-show status

# In[ ]:


# Distribution of Gender
gender_hist = sns.histplot(df.gender)

plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Gender Distribution")
plt.show()


# In[ ]:


df["gender"].value_counts(normalize=True).mul(100)


# almost 2/3rd of the population is female. lets now look at the No-show rate amongst the genders.

# In[ ]:


gender_counts = pd.crosstab(
    df["gender"],
    df["no_show_int"]
)

gender_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("Gender group")
plt.ylabel("Count")
plt.title("Attendance vs No-show by Gender")
plt.legend(["Attended", "No-show"])
plt.xticks(rotation=0)
plt.show()


# In[ ]:


df.groupby("gender")["no_show_int"].mean().mul(100)


# The distribution of no-show between males and females is relatively equal. Telling us that gender alone is not a perfect predictor of attendance. 

# ## SMS Received

# What is the distribution of receiving a notification and can it be related to the attendance. 

# In[ ]:


# Distribution of sms_received
sms_hist = sns.countplot(data = df, x = "sms_received")

plt.xlabel("Received SMS")
plt.ylabel("Count")
plt.title("SMS Received Distribution")
plt.show()


# In[ ]:


df["sms_received"].value_counts(normalize=True).mul(100)


# Just like the gender the split is around 2/3 to 1/3. Lets again look at the no-show rates between them.

# In[ ]:


sms_counts = pd.crosstab(
    df["sms_received"],
    df["no_show_int"]
)

sms_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("SMS Received")
plt.ylabel("Count")
plt.title("Attendance vs No-show by SMS Received")
plt.legend(["Attended", "No-show"])
plt.xticks(rotation=0)
plt.show()


# In[ ]:


df.groupby("sms_received")["no_show_int"].mean().mul(100)


# Interestingly it seems like patients who received the SMS have a higher no-show rate. 
# 
# As i suspect the chance of younger patients having set up the SMS reminder is higher than older patients i will now take quick look if this could be a partial cause of the difference

# In[ ]:


df.groupby("age_group")["sms_received"].mean().mul(100)


# It seems like my suspicion was incorrect and the SMS received i relatively eual between groups 20 to 70, only dropping 4 percent in the other groups. 

# ### Length of time between appointment and time of scheduling (wait_time)

# In[ ]:


# Distribution of wait_time
wait_time_hist = sns.histplot(df.wait_time)
wait_time_hist.plot(
    kind="hist",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("Time between Scheduling and Appointment")
plt.ylabel("Count")
plt.title("wait_time Distribution")
plt.ylim(0, 7500) # as 0 days has a count of almost 40.000 we cut it of here to see the distribution better
plt.show()


# amount of appointments scheduled more than 2 months seems very low. Lets take a look at the amount over 61 days. 

# In[ ]:


df.loc[df["wait_time"] > 61, "wait_time"].value_counts().sort_index()
# Distribution of wait_time
wait_time_hist = sns.histplot(df.loc[df["wait_time"] > 61, "wait_time"])
wait_time_hist.plot(
    kind="hist",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("Time between Scheduling and Appointment")
plt.ylabel("Count")
plt.title("wait_time over 61 Distribution")


# Results does not seem unreasonable as some appointments might just be scheduled far in the future. from the max value of 179 days we saw earlier it is around 6 months so that seems reasonable to me. 

# Next lets look at the behaviour of no-show over the waiting time. 
# 
# First we will do a scatterplot to see if the percentage of patients who no-show increases or decreases over a longer waiting time. 

# In[ ]:


# Calculate no-show rate per wait_time
wait_time_rate = (
    df.groupby("wait_time")["no_show_int"]
    .mean()
    .mul(100)
    .reset_index(name="no_show_rate")
)

# Plot
plt.figure(figsize=(12, 6))

sns.scatterplot(
    data=wait_time_rate,
    x="wait_time",
    y="no_show_rate"
)

plt.xlabel("Wait Time (days)")
plt.ylabel("No-show Rate (%)")
plt.title("No-show Rate by Wait Time")

plt.show()


# Conclusion for this figure seems that the no-show rate seems to be relatively stable over the wait time, hoever the longer the wait the less data we have so the scatterplot goes up and down allot. 

# We can also make a box plot to check the distribution of the wait time among subjects who no-showed vs subjects that attended. 

# In[ ]:


plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="no_show",
    y="wait_time"
)

plt.xlabel("No-show")
plt.ylabel("Wait Time (days)")
plt.title("Wait Time by No-show Status")

plt.xticks([0, 1], ["Showed up", "No-show"])

plt.show()


# Conclustion subjects that were No-shows generally had a longer waiting time than people who did show up. 
# 
# I can imagine that this is influenced by the large amount of subjects with 0 days. Lets remove those from this boxplot and check again. 

# In[ ]:


plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df[df["wait_time"] > 0],
    x="no_show",
    y="wait_time"
)

plt.xlabel("No-show")
plt.ylabel("Wait Time (days)")
plt.title("Wait Time by No-show Status (excluding same-day appointments)")

plt.xticks([0, 1], ["Showed up", "No-show"])

plt.show()


# The difference is less stark between the two groups but is still present, i think this wait time could be a good predictor. 

# ### general health 
# for each of the 4 health categories do a quick check if there is anything that grabs the attention. 
# todo: overview of hipertension, diabetes, alcoholism, handcap .

# In[ ]:


hipertension_counts = pd.crosstab(
    df["hipertension"],
    df["no_show_int"]
)

hipertension_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("hipertension group")
plt.ylabel("Count")
plt.title("Attendance vs No-show by hipertension")
plt.legend(["Attended", "No-show"])
plt.xticks(rotation=0)
plt.show()


# In[ ]:


diabetes_counts = pd.crosstab(
    df["diabetes"],
    df["no_show_int"]
)

diabetes_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("diabetes group")
plt.ylabel("Count")
plt.title("Attendance vs No-show by diabetes")
plt.legend(["Attended", "No-show"])
plt.xticks(rotation=0)
plt.show()


# In[ ]:


alcoholism_counts = pd.crosstab(
    df["alcoholism"],
    df["no_show_int"]
)

alcoholism_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("alcoholism group")
plt.ylabel("Count")
plt.title("Attendance vs No-show by alcoholism")
plt.legend(["Attended", "No-show"])
plt.xticks(rotation=0)
plt.show()


# In[ ]:


handcap_counts = pd.crosstab(
    df["handcap"],
    df["no_show_int"]
)

handcap_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("handcap group")
plt.ylabel("Count")
plt.title("Attendance vs No-show by handcap")
plt.legend(["Attended", "No-show"])
plt.xticks(rotation=0)
plt.show()


# Has this is hard to see for the smaller values we will normalize it to percentages

# In[ ]:


handcap_percent = pd.crosstab(
    df["handcap"],
    df["no_show_int"],
    normalize="index"
) * 100

handcap_percent.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("Handcap group")
plt.ylabel("Percentage (%)")
plt.title("Attendance vs No-show by Handcap")
plt.legend(["Attended", "No-show"])
plt.xticks(rotation=0)
plt.ylim(0, 100)

plt.show()


# No category seems to be overrepresented for no-show. 

# ### neighbourhood
# 
# 

# In[ ]:


neighbourhood_counts = pd.crosstab(
    df["neighbourhood"],
    df["no_show_int"]
)

neighbourhood_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("neighbourhood")
plt.ylabel("Count")
plt.title("Attendance vs No-show by neighbourhood")
plt.legend(["Attended", "No-show"])
plt.xticks(rotation=90)
plt.show()


# Just the count does not really a good view of any neighbourhood stands out one way or another. 

# In[ ]:


neighbourhood_percent = pd.crosstab(
    df["neighbourhood"],
    df["no_show_int"],
    normalize="index"
) * 100

neighbourhood_percent.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("neighbourhood group")
plt.ylabel("Percentage (%)")
plt.title("Attendance vs No-show by neighbourhood")
plt.legend(["Attended", "No-show"])
plt.xticks(rotation=90)
plt.ylim(0, 100)

plt.show()


# No real standouts percentage wise except 2 which looking at the count graph have only a couple counts. 

# ## Feature importance
# 
# ### split the datasets

# In[ ]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

#reset the index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

#isolate the target
y_train = df_train.no_show_int.values
y_val = df_val.no_show_int.values
y_test = df_test.no_show_int.values

#delete the no_show columns
del df_train['no_show_int']
del df_val['no_show_int']
del df_test['no_show_int']

del df_train['no_show']
del df_val['no_show']
del df_test['no_show']


# ### Risk

# In[ ]:


# EDA, feature importance analysis


global_noshow = df_full_train.no_show_int.mean()

for col in categorical:
    print(col)
    df_group = df_full_train.groupby(col).no_show_int.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_noshow
    df_group['risk'] = df_group['mean'] / global_noshow
    display(df_group.round(3))
    print()
    print()



# In[ ]:





# Going over all the tables some of them stand out, i will quickly go over all of them to create a short overview
# 
# **gender:**
# 
# The difference in no-show risk between Male and Female is very small so probably not an important feature
# 
# **age_group:**
# 
# The difference in no-show risk between the age groups range from 0.737 to 1.261, it might be important feature
# 
# **sms_received:**
# 
# The difference in no-show risk between having received text message or not goes from 0.831 for no message to 1.356 for receiving a message. 
# counterintuitively indicating that receiving a message is negatively influencing the attendance. 
# 
# **hipertension:**
# 
# Not having hipertension has a no-show risk of 1.035, while having hipertension gives a risk of 0.857, indicating that having hipertension is an indicator for showing up while not having hipertension does not necessarily influence the attendance
# 
# **diabetes:**
# 
# Not having diabetes has a no show risk of 1.009, while having diabetes has as risk of 0.889, so not having diabetes is not a good indicator while having diabetes can be a good indicator for showing up. 
# 
# **alcoholism:**
# 
# Not having alcoholism has a risk of 1.000, and having alcoholism has a risk of 0.985, so not having alcoholism is not an indicator while having alcoholism can be an indicator. 
# 
# **handcap:**
# 
# Not having a handicap gives a risk of 1.002, handcap severity of 1 or 2 has a negative effect on no-show while 3 and 4 have a positive effect on noshow. 
#     it can be a good indicator if any handcap is prensent. however the higher categories can give false information due to the low amount of subjects. 
# 
# **scholarship:**
# 
# Not having a scholarship has a risk of 0.980, indicating it has no influence while having a scholarship has a risk of 1.184 indicating it can have an influence. 
# 
# **neighbourhood:**
# 
# neighbourhood is a bit harder to judge, however some neighbourhoods seem to have an influence on the no-show rate. 
# 

# ## Mutual information

# In[ ]:


def mutual_info_noshow_score(categorical):
    return mutual_info_score(categorical, df_full_train.no_show_int)


mi = df_full_train[categorical].apply(mutual_info_noshow_score)
mi.sort_values(ascending=False).round(6)


# sms_reveived seems to be the most important variable, followed by age group and neighbourhood. 
# all other variables seem to be much less important. 

# ## Correlation

# In[ ]:


df_full_train[numerical].corrwith(df_full_train.no_show_int)


# - age has a low negative correlation to no-show. the older they are the less likely to no-show
# - wait_time has a low-to-medium positive correlation. The longer the wait time the more likely they are to no-show.

# ### One-hot encoding

# Create binary columns where necessary

# In[ ]:


dicts = df[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')

X_train = dv.fit_transform(train_dict)

# dv.get_feature_names_out() # uncomment if you want to see the created columns

val_dict = df_val[categorical + numerical].to_dict(orient='records')
test_dict = df_test[categorical + numerical].to_dict(orient='records')

X_val = dv.transform(val_dict)
X_test = dv.transform(test_dict)

X_train.shape



# The matrix now has 105 features

# ## Logistic regression model

# In[ ]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# #### how well did the model produce the outcome
# 

# In[ ]:


y_pred = model.predict_proba(X_val)[:, 1]

no_show_decision = (y_pred >= 0.5)

(y_val == no_show_decision).mean()




# 79.8% of the time the model predicted the attendance correctly.

# In[ ]:


# model.intercept_[0]
# model.coef_[0].round(3)


# <!-- Lets take a look at the coefficients -->

# In[ ]:


# give a name to each weight
# dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))


# In[ ]:


# commented out the small interpretation helper, no conclusions were made using this just used for a quick view
# small = ['age_group', 'scholarship', 'sms_received']

# dicts_train_small = df_train[small].to_dict(orient='records')
# dicts_val_small = df_val[small].to_dict(orient='records')

# dv_small = DictVectorizer(sparse=False)
# dv_small.fit(dicts_train_small)

# dv_small.get_feature_names_out()

# X_train_small = dv_small.transform(dicts_train_small)

# model_small = LogisticRegression(solver='lbfgs')
# model_small.fit(X_train_small, y_train)

# w = model_small.coef_[0]
# dict(zip(dv_small.get_feature_names_out(), w.round(3)))


# ### First model without tuning and its accuracy
# 

# In[ ]:


# final model
dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

y_full_train = df_full_train.no_show_int.values

model_basic = LogisticRegression(max_iter=1000) # increased the max_iter as it failed to converge
model_basic.fit(X_full_train, y_full_train)

# checking on the test set
dicts_test = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(dicts_test)

y_pred_basic = model_basic.predict_proba(X_test)[:, 1]
no_show_decision = (y_pred_basic >= 0.5)
(no_show_decision == y_test).mean()


# The accuracy on the test set is 79.6%, on the validation dataset we had 79.8%. The two numbers are close, the model preforms as good on unseen data as on the data it was tuned with. 
# No indication of overfitting. 
# 
# 

# In[ ]:


# from sklearn.metrics import accuracy_score

# y_pred = model.predict_proba(X_val)[:, 1]

# thresholds = np.linspace(0, 1, 21)

# scores = []

# for t in thresholds:
#     score = accuracy_score(y_val, y_pred >= t)
#     print('%.2f %.3f' % (t, score)) # make this a comment if you prefer to see the plot
#     scores.append(score)


# plt.plot(thresholds, scores, marker="o") # uncomment if you prefer to see the result in a plot


# Conclusion seems to be that a threshold at about 0.5 is almost the best it can do, setting the threshold to 65 only increases the accuracy by 0.2 %

# In[ ]:


# from collections import Counter

# Counter(y_pred >= 1.0)

# 1 - y_val.mean()


# result is 80% so a model that does not look at the data scores 80%, our model is thus not better than just predicting everyone attend. (no_show = 0)

# ### Confustion table

# In[ ]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)

t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)

tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()

confusion_matrix = pd.DataFrame(
    [
        [tn, fp],
        [fn, tp]
    ],
    index=["Actual Show", "Actual No-show"],
    columns=["Predicted Show", "Predicted No-show"]
)

normalized_confusion_matrix = (
    confusion_matrix / confusion_matrix.to_numpy().sum()*100
).round(1) 

normalized_confusion_matrix


# So 79.4% of all customers are true negatives, 0.6% false positives, 19.6% false negatives and 0.4% true positives.

# In[ ]:


p = tp / (tp + fp)
r = tp / (tp + fn)

pd.DataFrame(
    [
        [(p*100).round(1)],
        [(r*100).round(1)]
    ],
    index=["Precistion", "Recall"],
    columns=["%"]
)



# Precision of around 38.4% and a recall of 2%.
# 
# so for all no-shows we only correctly flagged 2%

# ### ROC

# In[ ]:


tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

pd.DataFrame(
    [
        [(tpr*100).round(1)],
        [(fpr*100).round(1)]
    ],
    index=["true positive rate", "true positive rate"],
    columns=["%"]
)


# Now for all thresholds

# In[ ]:


scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    scores.append((t, tp, fp, fn, tn))

columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
plt.legend()


# In[ ]:


plt.figure(figsize=(5, 5))

plt.plot(df_scores.fpr, df_scores.tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# ### AUC

# In[ ]:


from sklearn.metrics import roc_auc_score

roc_auc_score(y_val, y_pred)


# The area under the curve is 0.66, which we can state is relatively poor. The model clearly needs to be trained. 

# ### parameter tuning
# 

# In[ ]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[ ]:


from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=1)


# In[ ]:


from tqdm.auto import tqdm

n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.no_show_int.values
        y_val = df_val.no_show_int.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# No strong sensitivity to C here, the models get slightly worse with a higher C. 
# recomendation is thus to set the C to 0.001

# Final logistic model is thus:

# In[ ]:


dv, model = train(df_full_train, df_full_train.no_show_int.values, C=0.001)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc


# ## Descision tree-based model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text

dicts_train = df_train[categorical + numerical].to_dict(orient="records")
dicts_val = df_val[categorical + numerical].to_dict(orient="records")

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(dicts_train)
X_val = dv.transform(dicts_val)


dt = DecisionTreeClassifier(
    max_depth=3, # without a max dept the AUC was 0.571 for validation and 0.996 for training data so it was overfitting. 
)

dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)
print('train:', auc)

y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print('val:', auc)


# training and validation AUC are almost equal, the model is generalizable over the split dataset. 

# In[ ]:


# uncomment if you would like to see the decision tree
# print(export_text(dt, feature_names=list(dv.get_feature_names_out())))


# ### parameter tuning
# 

# #### Tune max_depth

# In[ ]:


# same is in the course we start with adjusting the max_depth
for depth in [1, 2, 3, 4, 5, 6, 10, 15, 20, None]:

    dt = DecisionTreeClassifier(
        max_depth=depth
    )

    dt.fit(X_train, y_train)

    y_pred = dt.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_pred)

    print(depth, round(auc, 3))


# Depths 4, 5, and 6 give the best results with a AUC of about 0.725

# #### Tune min_samples_leaf

# In[ ]:


scores = []

for depth in [4, 5, 6]:
    for leaf_size in [1, 5, 10, 15, 20, 50, 100]:

        dt = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf_size,
            random_state=1
        )

        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((depth, leaf_size, auc))


columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)


# AUC valuees are almost equal regardless of min_samples_leaf.
# depths 4 and 5 see no difference in AUC uppon changing the min_samples_leaf.
# depth 6 is slightly better for min_samples_leaf = 5 

# Lets check if anything interesting happens if we do this again for all depths.

# In[ ]:


scores = []

for depth in [1, 2, 3, 4, 5, 6, 10, 15, 20, None]:
    for leaf_size in [1, 5, 10, 15, 20, 50, 100]:

        dt = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf_size,
            random_state=1
        )

        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((depth, leaf_size, auc))


columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)


# No additional combinations stand out as improvements, however interestingly depth = 10 with min_samples_leaf = 100 performs equally to depth = 6 + min_samples_leaf = 5
# 
# Next lets check out the tree with the tuned parameters

# In[ ]:


dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5)
dt.fit(X_train, y_train)

# print(export_text(dt, feature_names=list(dv.get_feature_names_out())))


# ## Random forest-based model

# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

scores = []

for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    scores.append((n, auc))

df_scores = pd.DataFrame(scores, columns=['n_estimators', 'auc'])
plt.plot(df_scores.n_estimators, df_scores.auc)


# Adding more than a 100 trees does little to the improve the AUC of the model

# #### Tuning max_depth

# In[ ]:


scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))


columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth == d]

    plt.plot(df_subset.n_estimators, df_subset.auc,
             label='max_depth=%d' % d)

plt.legend()


# Higher depth gives a decent improvement, as higher depth might improve even more so we will do this again with 15, 20 and 25

# In[ ]:


scores = []

for d in [15, 20, 25]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))


columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

for d in [15, 20, 25]:
    df_subset = df_scores[df_scores.max_depth == d]

    plt.plot(df_subset.n_estimators, df_subset.auc,
             label='max_depth=%d' % d)

plt.legend()


# A depth of 20 gives our best results, it starts to deteriorate at 25. 

# In[ ]:


max_depth = 20


# #### Tuning min_samples_leaf

# In[ ]:


scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=s,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))

columns = ['min_samples_leaf', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 3, 5, 10, 50]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]

    plt.plot(df_subset.n_estimators, df_subset.auc,
             color=col,
             label='min_samples_leaf=%d' % s)

plt.legend()


# Curves 1 to 5 are close together but diverge after a n_estimators of 75. 
# we will pick a leaf of 1 as it is slightly higher.

# In[ ]:


scores = []

for s in [1, 2, 3, 4, 5]:
    for n in range(10, 500, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=s,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))

columns = ['min_samples_leaf', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 2, 3, 4, 5]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]

    plt.plot(df_subset.n_estimators, df_subset.auc,
             color=col,
             label='min_samples_leaf=%d' % s)

plt.legend()


# Curves ....

# In[ ]:


rf = RandomForestClassifier(n_estimators=200,
                            max_depth=max_depth,
                            min_samples_leaf=1,
                            random_state=1)
rf.fit(X_train, y_train)


# ## Export model training logic to script

# 
