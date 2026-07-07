
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# %%

df = pd.read_csv("loan_approval_data.csv")
# %%
df.head()
cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category"
]

print(df[cols].nunique())

# %%
df.info()
# %%
df.isnull().sum()
# %%
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["float64"]).columns

# %%
categorical_cols
# %%

numerical_cols
# %%
from sklearn.impute import SimpleImputer

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols]=num_imp.fit_transform(df[numerical_cols])
# %%
cato_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cato_imp.fit_transform(df[categorical_cols])
# %%

df.head()
df.isnull().sum()

# %%
# EDA
classes_count = df["Loan_Approved"].value_counts()
plt.pie(classes_count,labels=["No","Yes"],autopct="%1.1f%%")
plt.title("Is loan approved or not?")

# %%
gender_cnt = df["Gender"].value_counts()
ax = sns.barplot(gender_cnt)
ax.bar_label(ax.containers[0])


edu_cnt = df["Gender"].value_counts()
ax = sns.barplot(edu_cnt)
ax.bar_label(ax.containers[0])


# %%
# analyse income

sns.histplot(
    data=df,
    x="Applicant_Income",
    bins=20
)

# %%


sns.histplot(
    data=df,
    x= "Coapplicant_Income",
    bins=20
)
# %%
# outliers

sns.boxplot(data=df,x="Loan_Approved",
            y="Applicant_Income")
# %%

fig,axes=plt.subplots(2,2)


sns.boxplot(ax=axes[0,0],data=df,x="Loan_Approved",y="Applicant_Income")
sns.boxplot(ax=axes[0,1],data=df,x="Loan_Approved",y="Credit_Score")
sns.boxplot(ax=axes[1,0],data=df,x="Loan_Approved",y="DTI_Ratio")
sns.boxplot(ax=axes[1,1],data=df,x="Loan_Approved",y="Savings")

plt.tight_layout()

# %%
# credit and approve

sns.histplot(
    data=df,
    x="Credit_Score",
    bins=20,
    multiple="dodge",
    hue="Loan_Approved"

)
# %%
# remove applicant id

df = df.drop("Applicant_ID",axis=1)
df.head()
# %%

# ab encoding kro
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# %%
df.info()
# %%
cols=["Employment_Status","  Marital_Status ","Loan_Purpose ","Property_Area ","Gender","Employer_Category 47"]
# %%
le = LabelEncoder()
df["Education_Level"] = le.fit_transform( df["Education_Level"])
df["Loan_Approved"]= le.fit_transform(df["Loan_Approved"])

# %%
df.head()


# %%
cols=["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
ohe = OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore")
encoded = ohe.fit_transform(df[cols])

encoded_df=pd.DataFrame(encoded,columns=ohe.get_feature_names_out(cols),index=df.index)

df=pd.concat([df.drop(columns=cols),encoded_df],axis=1)
             

# %% 
encoded_df.head()
# %%
nums_col = df.select_dtypes(include="number")
corr_matrix = nums_col.corr()


# %%
nums_col.corr()["Loan_Approved"].sort_values(ascending=False)
# %%
sns.heatmap(corr_matrix,annot=True,fmt=".2f",cmap="coolwarm")
# %%
x = df.drop("Loan_Approved",axis=1)
y = df["Loan_Approved"]
x.head()
# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# %%
# model training
# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

log_model = LogisticRegression()
log_model.fit(x_train_scaled,y_train)


# %%
y_pred = log_model.predict(x_test_scaled)
# %%
print("precision=",precision_score(y_test,y_pred))
print("accuracy=",accuracy_score(y_test,y_pred))
print("recall=",recall_score(y_test,y_pred))
print("f1=",f1_score(y_test,y_pred))
# %%
# knn clasifier
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train_scaled,y_train)
y_pred = knn_model.predict(x_test_scaled)
print("precision=",precision_score(y_test,y_pred))
print("accuracy=",accuracy_score(y_test,y_pred))
print("recall=",recall_score(y_test,y_pred))
print("f1=",f1_score(y_test,y_pred))


# %%

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(x_train_scaled,y_train)
y_pred = nb_model.predict(x_test_scaled)
print("precision=",precision_score(y_test,y_pred))
print("accuracy=",accuracy_score(y_test,y_pred))
print("recall=",recall_score(y_test,y_pred))
print("f1=",f1_score(y_test,y_pred))

# %%
# inthis problem main aim is to improve score in which false negatives get approved means we have to train model such that a false customer should not get load approved anyhow
# in this case precision score matter so naive_bayes is winner
# %%
# feature engineering
df["DTI_Ratio_sq"] = df["DTI_Ratio"]**2
df["Credit_Score_sq"] = df["Credit_Score"]**2
# for handling skew data
# df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])
x= df.drop(columns=["Loan_Approved","DTI_Ratio","Credit_Score"])
y= df["Loan_Approved"]


# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)

# %%
# scaling
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
nb_model.fit(x_train_scaled,y_train)
# %%
y_pred = nb_model.predict(x_test_scaled)
print("precision=",precision_score(y_test,y_pred))
print("accuracy=",accuracy_score(y_test,y_pred))
print("recall=",recall_score(y_test,y_pred))
print("f1=",f1_score(y_test,y_pred))

# %%
