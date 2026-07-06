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
df.head()
# %%


# %%
