# %%
print("challlenge accepted")
# %%
import pandas as pd
# %%
train_df = pd.read_csv('Train_Keystroke.csv')
# %%
train_df.head()

# %%
test_df = pd.read_csv('Test_Keystroke.csv')
test_df.head()
# %%
print("----User appearances----\n",train_df.UserID.value_counts()\
    ,"\n ----Distinct users---- \n",train_df.UserID.unique()\
    ,"\n ----Number of distinct users----\n", len(train_df.UserID.unique()) )















# %%
#maximum press time character per user
max_press = train_df.loc[:,~train_df.columns.str.startswith('release')]\
    .groupby("UserID").max("press*").idxmax(axis=1)
max_press
# %%
#maximum press time character per user
max_rel = train_df.loc[:,~train_df.columns.str.startswith('press')]\
    .groupby("UserID").max("release*").idxmax(axis=1)
max_rel

# %%
print(max_press.unique(), max_rel.unique())

# %%
#take differences between consecutive columns to check if presses 
# and releases are monotonous

print(train_df.head())
rel_dif = train_df.loc[:,train_df.columns.str.startswith('release')].diff(axis=1)
print("Any negative dif in realeases?",(rel_dif<0).any().any())

press_dif = train_df.loc[:,train_df.columns.str.startswith('press')].diff(axis=1)
print("Any negative dif in presses?",(rel_dif<0).any().any())

#Check if the biggest differences are noticed at a specific point
rel_dif['UserID'] = train_df['UserID'].values
press_dif['UserID'] = train_df['UserID'].values

print("#Columns with biggest dif from previous in releases for user's worst",\
    len(rel_dif.groupby("UserID").max("releases*").idxmax(axis=1).unique()))
print("#Columns with biggest dif from previous in presses for user's worst",\
    len(press_dif.groupby("UserID").max("press*").idxmax(axis=1).unique()))

print("#Columns with biggest dif from previous in releases",\
    len(rel_dif.idxmax(axis=1).unique()))
print("#Columns with biggest dif from previous in presses",\
    len(press_dif.idxmax(axis=1).unique()))

#find users' worst performance's iteration
rel_dif['increment'] = rel_dif.groupby('UserID').cumcount() + 1
press_dif['increment'] = press_dif.groupby('UserID').cumcount() + 1

print("Users' worst iteration",\
    rel_dif.groupby("UserID").max("releases*").idxmax(axis=0))
print("Users' worst iteration",\
    press_dif.groupby("UserID").max("press*").idxmax(axis=0))


# %%
#find users' worst performance's iteration
#valuable only if users' iterarations are stored in order
rel_dif['increment'] = rel_dif.groupby('UserID').cumcount() + 1
press_dif['increment'] = press_dif.groupby('UserID').cumcount() + 1

print("Users' worst iteration on releasing",\
    rel_dif.groupby("UserID").max("releases*").increment)
print("Users' worst iteration on pressing",\
    press_dif.groupby("UserID").max("press*").increment)












# %%
train_df.head()
#they are timestamps so i'll take the differences between releases of a button 
# and releases of the next one to calculate the time that takes the user to find
#  the next button but also the difference between press and release to calculate
#  the time taking to release a button.

# %%
#duration = release - press 
duration_df = pd.DataFrame()
for i in range(13):
    duration_df["duration-"+str(i)] = train_df["release-"+str(i)]-train_df["press-"+str(i)]
duration_df
# %%
#delay = difference between pressing consecutive keys, hence time taken from one
#to another
delay_df = pd.DataFrame()
for i in range(12):
    delay_df["delay"+str(i)+"-"+str(i+1)] = train_df["press-"+str(i+1)]-train_df["press-"+str(i)]
delay_df

# %%
#add UserID in the new dataframes
duration_df['UserID'] = train_df['UserID'].values
delay_df['UserID'] = train_df['UserID'].values
duration_df
# %%
#max delay and duration index for each user
import matplotlib.pyplot as plt
max_dur = duration_df.groupby("UserID").max("duration-*").idxmax(axis=1)
max_del = delay_df.groupby("UserID").max("delay*").idxmax(axis=1)
print(max_dur)
ax_dur = max_dur.value_counts().plot.bar()
plt.show()
ax_del = max_del.value_counts().plot.bar()
plt.show()

# %%
#find users' worst performance's iteration
#valuable only if users' iterarations are stored in order
duration_df['increment'] = duration_df.groupby('UserID').cumcount() + 1
delay_df['increment'] = delay_df.groupby('UserID').cumcount() + 1

print("Users' worst iteration duration wise \n",\
    duration_df.groupby("UserID").max("duration-*").increment)
print("Users' worst iteration pressing wise \n",\
    delay_df.groupby("UserID").max("delay*").increment)

# %%
#check mean per iteration
print(duration_df.groupby("increment").mean("duration-*").mean(axis=1)\
,delay_df.groupby("increment").mean("delay*").mean(axis=1))
#-->start kind off slow, then find some rythm but eventually get tired


# %%
duration_df.groupby("UserID").mean("duration-*").plot.line(legend=False\
    ,figsize=(50,50))
plt.show()
delay_df.groupby("UserID").mean("delay*").plot.line(legend=False\
    ,figsize=(50,50))


# %%
#check how each user performs over time (their mean of over all letters)

duration_df["mean"] = duration_df.loc[:,[c for c in \
    duration_df.columns if c!= "UserID" and c!="increment"]].mean(axis=1)
dur_df_incr = duration_df[["UserID","increment","mean"]]
dur_df_incr.set_index('UserID').T.plot.line(legend=False, figsize=(20,30))


delay_df["mean"] = delay_df.loc[:,[c for c in \
    delay_df.columns if c!= "UserID" and c!="increment"]].mean(axis=1)
del_df_incr = delay_df[["UserID","increment","mean"]]
del_df_incr.set_index('UserID').T.plot.line(legend=False, figsize=(20,30))

# %%
duration_df.describe()
# %%
delay_df.describe()




# %%
#FEATURE ENGINEERING
for i in range(13):
    train_df["duration-"+str(i)] = train_df["release-"+str(i)]-train_df["press-"+str(i)]
for i in range(12):
    train_df["delay"+str(i)+"-"+str(i+1)] = train_df["press-"+str(i+1)]-train_df["press-"+str(i)]
for i in range(12):
    train_df["overlap: press"+str(i+1)+"-"+"release"+str(i)] = train_df["press-"+str(i+1)]-train_df["release-"+str(i)]    
train_df.head()

# %%
train_df["mean_rel"] = train_df[[col for col in train_df.columns \
    if col.startswith('release')]].mean(axis=1)
train_df["mean_press"] = train_df[[col for col in train_df.columns \
    if col.startswith('press')]].mean(axis=1)
train_df["mean_dur"] = train_df[[col for col in train_df.columns \
    if col.startswith('duration')]].mean(axis=1)    
train_df["mean_del"] = train_df[[col for col in train_df.columns \
    if col.startswith('delay')]].mean(axis=1)
train_df["mean_overlap"] = train_df[[col for col in train_df.columns \
    if col.startswith('overlap')]].mean(axis=1)
train_df.head()

# %%
train_df["std_rel"] = train_df[[col for col in train_df.columns \
    if col.startswith('release')]].std(axis=1)
train_df["std_press"] = train_df[[col for col in train_df.columns \
    if col.startswith('press')]].std(axis=1)
train_df["std_dur"] = train_df[[col for col in train_df.columns \
    if col.startswith('duration')]].std(axis=1)    
train_df["std_del"] = train_df[[col for col in train_df.columns \
    if col.startswith('delay')]].std(axis=1)
train_df["std_overlap"] = train_df[[col for col in train_df.columns \
    if col.startswith('overlap')]].std(axis=1)
train_df.head()
# %%
train_df["total_dur"] = train_df[[col for col in train_df.columns \
    if col.startswith('duration')]].sum(axis=1)
train_df["total_del"] = train_df[[col for col in train_df.columns \
    if col.startswith('delay')]].sum(axis=1)
train_df["del-dur"] = train_df["total_del"] - train_df["total_dur"]    
train_df.head()

# %%
train_df["max_dur"] = train_df[[col for col in train_df.columns \
    if col.startswith('duration')]].max(axis=1)
train_df["max_del"] = train_df[[col for col in train_df.columns \
    if col.startswith('delay')]].max(axis=1)
train_df["max_del-dur"] = train_df["max_del"] - train_df["max_dur"]    
train_df.head()

# %%
train_df.columns

initial_cols = ['press-0', 'release-0', 'press-1', 'release-1', 'press-2', 'release-2',
       'press-3', 'release-3', 'press-4', 'release-4', 'press-5', 'release-5',
       'press-6', 'release-6', 'press-7', 'release-7', 'press-8', 'release-8',
       'press-9', 'release-9', 'press-10', 'release-10', 'press-11',
       'release-11', 'press-12', 'release-12']

initial_cols

# %%
train_df.drop(initial_cols, axis=1, inplace=True)
# %%
print(train_df.columns)
train_df.head()











# %%
#MODELS
import numpy as np
from packaging import version
import sklearn
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
#NO PIPELINE: no scaler, imputer needed, no categorical data  

# %%
features = train_df.loc[:, train_df.columns != 'UserID']
target = train_df["UserID"].values
target = target.astype(int)
target
# %%
models=[svm.SVC(kernel='rbf',random_state=1),RandomForestClassifier(random_state=1)]
scores=[]
kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
shuf=ShuffleSplit(n_splits=5, random_state=1)
for m in models:
    res=m.fit(features,target)
    scores2=cross_validate(res,features,target,scoring='accuracy',cv=kfold) #cv=shuf is also a choice
    f_s2=scores2['test_score'].mean()
    scores.append(f_s2)
scores=np.array(scores)
scores
#svm needs normalization, tuning and appropriate kernel, here linear fits 
# best after a fast/small experimentation because of the structure of the data


# %%
#TUNING RF
from sklearn.model_selection import GridSearchCV 

modelo = RandomForestClassifier(random_state=1)
estim=list(np.linspace(100,1500,4).astype(int)) #trees
max_f=[0.05,0.1,'auto'] #features

params= {'n_estimators':estim , 'max_features':max_f}
res=GridSearchCV(modelo, param_grid=params,cv=kfold,scoring='accuracy', \
    n_jobs=-1).fit(features, target)
scores = res.cv_results_['mean_test_score'].reshape(len(params['n_estimators']),\
                                len(params['max_features'])).T


# %%
import seaborn as sns
def heatmap(columns, rows, scores):
    """ Simple heatmap.
    Keyword arguments:
    columns -- list of options in the columns
    rows -- list of options in the rows
    scores -- numpy array of shape (#rows, #columns) of scores
    """
    df = pd.DataFrame(scores, index=rows, columns=columns)
    sns.heatmap(df, cmap='Greens', linewidths=0.5, annot=True, fmt=".3f")
heatmap(params['n_estimators'], params['max_features'], scores)

# %%
#TUNING RF with Randomized search
from sklearn.model_selection import RandomizedSearchCV 

modelo = RandomForestClassifier(random_state=1)
estim=list(np.linspace(100,1500,4).astype(int)) #trees
max_f=[0.05,0.1,'auto'] #features

params= {'n_estimators':estim , 'max_features':max_f}
res=RandomizedSearchCV(modelo, param_distributions=params,cv=kfold,scoring='accuracy', \
    n_jobs=-1).fit(features, target)
res.best_params_

#n_estimatprs: 1500, max_features: 0.1
# %%
#TUNED RF (random)
rf_rand = RandomForestClassifier(n_estimators=1500, max_features= 0.1,random_state=1)
rf_rand.fit(features,target)
scores_rand=cross_validate(rf_rand,features,target,scoring='accuracy',cv=kfold) #cv=shuf is also a choice
f_rand=scores_rand['test_score'].mean()
# %%
f_rand
#modelo.get_params().keys()










# %%
print(test_model.predict([features.iloc[0]]))
# %%
train_df.head()















# %%
d = {'col1': [1, 3, 4, 5], 'col2': [3, 4, 1, 2], 'ID': [1, 2, 1, 2]}
df = pd.DataFrame(data=d)
df

# %%
df.groupby("ID").max("col*").idxmax(axis=1)
# %%
df.values
# %%
rel_dif
# %%
rel_dif
# %%
