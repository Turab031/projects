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
