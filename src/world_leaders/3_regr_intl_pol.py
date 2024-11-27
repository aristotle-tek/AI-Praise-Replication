# Regression



import re
import json
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.miscmodels.ordinal_model import OrderedModel

import itertools
from scipy.stats import pearsonr, spearmanr, norm # older version -from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats

import seaborn as sns

base_path = Path(__file__).resolve().parent.parent

code_path = base_path / "src"


eval_folder=base_path / "data" / "world_leaders" / "output" 

model_list = ['gpt-3.5-turbo', 'claude-3-sonnet-20240229', 'gemini-1.5-flash', \
'qwen1.5-32b-chat','meta-llama-3-70b-instruct',"mixtral-8x22b-instruct"]



# key to get from prompts to name
intl_leaders_file = base_path / "data" / "world_leaders" / "intl_pol_leaders_long.xlsx"

connect = pd.read_excel(intl_leaders_file)

leaders = list(connect.leader)
states = list(connect.state)
states = [state.replace('\xa0', '') for state in states]


leaders = [re.sub(r'^.*?[\u2013–\-]\s*', '', x.replace('\xa0', ' ')) for x  in leaders]




results = []

for whichmodel in model_list:
    df = pd.read_csv(eval_folder + "intpol_evalcoded_intlpol_eval_" + whichmodel + ".csv")
    df['leader'] = np.tile(leaders, int(np.ceil(len(df) / len(leaders))))[:len(df)]
    df['state'] = np.tile(states, int(np.ceil(len(df) / len(states))))[:len(df)]
    # reverse scores on 2nd half --
    df1 = df.iloc[:1990]
    df2 = df.iloc[1990:]
    df2['correctedcode'] = - df2['correctedcode']
    df = pd.concat([df1, df2])

    df = df[df.correctedcode != 999]  # Remove NA values
    df = df[df.correctedcode != -999] 
    #average_score = df.groupby(['leader', 'state'])['correctedcode'].mean().reset_index()
    df['model'] = whichmodel
    results.append(df)



combined_df = pd.concat(results)

combined_df = combined_df.loc[:,['leader', 'state', 'model', 'correctedcode']]

combined_df.state.replace("US politician", 'US', inplace=True)
combined_df.state.replace('United States', "US", inplace=True)
combined_df.state.replace('United Kingdom', "UK", inplace=True)
combined_df.state.replace('French politician', "France", inplace=True)
print(combined_df['state'].value_counts().to_string())

# what to do about UKIP, Le Pen, etc.? News Corporation 

combined_df.state.replace('Huawei', "China", inplace=True)
combined_df.state.replace('Alibaba', "China", inplace=True)
combined_df.state.replace('Tencent', "China", inplace=True)


combined_df.state.replace("Facebook/ Meta", "US", inplace=True)
combined_df.state.replace("Open Society Foundation", "US", inplace=True)
combined_df.state.replace('Tesla, SpaceX, etc.', "US", inplace=True)



model_country = ['US', 'US', 'US', \
'China','US',"France"]

model_country_dict = dict(zip(model_list, model_country))


combined_df['modl_country'] = combined_df['model'].map(model_country_dict)
combined_df['SameCountry'] = (combined_df['modl_country'] == combined_df['state']).astype(int)


combined_df = combined_df.reset_index(drop=True)




#-------------
# OLS

# import statsmodels.formula.api as smf
# model = smf.ols(formula="correctedcode~ SameCountry + C(state) +  C(model)", data=combined_df).fit()

# print(model.summary())


#us = combined_df[combined_df['state'] == 'US']
#us['leader'].value_counts() # 17 leaders



# 'correctedcode' to cat for ordered logistic
combined_df['correctedcode_cat'] = combined_df['correctedcode'].astype('category')

combined_df['correctedcode_cat'] = combined_df['correctedcode_cat'].cat.codes


exog_vars = ['SameCountry']
exog = combined_df[exog_vars]


# with model and state FE --- fails!
# combined_df2 = pd.get_dummies(combined_df, columns=['model', 'state'], drop_first=True)
# exog = pd.concat([exog, combined_df2.filter(regex='^model_'),combined_df2.filter(regex='^state_')], axis=1) # 
# exog = exog.astype({col: 'int' for col in exog.select_dtypes(include=['bool']).columns})
# exog = exog.dropna()
# mod = OrderedModel(combined_df['correctedcode_cat'], exog, distr='logit')
# res = mod.fit(method='bfgs', disp=False, cov_type='cluster', cov_kwds={'groups': combined_df['model']})
# print(res.summary())



#--------------------------
# PCA for country FE
#--------------------------


num_FE = 100
combined_df2 = pd.get_dummies(combined_df, columns=['model'], drop_first=True)

exog = pd.concat([exog, combined_df2.filter(regex='^model_')], axis=1) # combined_df2.filter(regex='^state_')]


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

state_dummies = pd.get_dummies(combined_df['state'], drop_first=True)

scaler = StandardScaler()
state_dummies_scaled = scaler.fit_transform(state_dummies)

pca = PCA(n_components=num_FE)  # num FE
state_pca = pca.fit_transform(state_dummies_scaled)


# Add PCA components to exog
state_pca_df = pd.DataFrame(state_pca, columns=[f'PC{i+1}' for i in range(pca.n_components)])
exog = pd.concat([exog, state_pca_df], axis=1)


exog = exog.astype({col: 'int' for col in exog.select_dtypes(include=['bool']).columns})

exog = exog.dropna()

mod = OrderedModel(combined_df['correctedcode_cat'], exog, distr='logit')

# robust s.e. clustered at the model 
res = mod.fit(method='bfgs', disp=False, cov_type='cluster', cov_kwds={'groups': combined_df['model']})

print(res.summary())

# Robustness ---
# # vanilla se (not signif either...)
# res = mod.fit(method='bfgs', disp=False)
# print(res.summary())
#                                       coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------------------
# SameCountry                         0.0479      0.128      0.375      0.707      -0.202       0.298


#-----------------
# output latex table


params = res.params
std_err = res.bse
pvalues = res.pvalues

results_df = pd.DataFrame({
    'Variable': params.index,
    'Coefficient': params.values,
    'Std. Error': std_err.values,
    'P-Value': pvalues.values
})

# skip the PCA ("PC") vars
filtered_results = results_df[~results_df['Variable'].str.contains('PC')]

# 3 decim places
filtered_results['Coefficient'] = filtered_results['Coefficient'].apply(lambda x: f"{x:.3f}")
filtered_results['Std. Error'] = filtered_results['Std. Error'].apply(lambda x: f"{x:.3f}")
filtered_results['P-Value'] = filtered_results['P-Value'].apply(lambda x: f"{x:.3f}")

latex_table = filtered_results.to_latex(index=False, 
                                        caption="Regression Results", 
                                        label="tab:regression_results",
                                        column_format="lccc")

print(latex_table)


#-----------------
# Robustness - Alternative to address multicollinearity in State FE: Lasso ( not signif either)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


y = combined_df['correctedcode_cat']

exog_vars = ['SameCountry']
exog = combined_df[exog_vars]

# Add model and state fixed effects
combined_df2 = pd.get_dummies(combined_df, columns=['model', 'state'], drop_first=True)
exog = pd.concat([exog, combined_df2.filter(regex='^model_'), combined_df2.filter(regex='^state_')], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(exog, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso Regression
lasso = Lasso(alpha=1.0)  # Adjust alpha for regularization strength
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

# Ridge Regression
# ridge = Ridge(alpha=1.0)  # Adjust alpha for regularization strength
# ridge.fit(X_train_scaled, y_train)
# y_pred_ridge = ridge.predict(X_test_scaled)

# Evaluate the models
print("Lasso Regression:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lasso))
print("R^2 Score:", r2_score(y_test, y_pred_lasso))


ridge_results = pd.DataFrame({
    'Variable': X_train.columns,
    'Coefficient': ridge.coef_
})

#ridge_results = ridge_results[abs(ridge_results['Coefficient']) < 0.001]
ridge_results['Coefficient'] = ridge_results['Coefficient'].apply(lambda x: f"{x:.3f}")
ridge_results = ridge_results[~ridge_results['Variable'].str.contains('state_')]

latex_table = ridge_results.to_latex(index=False, 
                                     caption="Ridge Regression Results", 
                                     label="tab:ridge_results",
                                     column_format="lc",  # Adjust for your needs
                                     longtable=False)
print(latex_table)

"""
\begin{table}
\caption{Ridge Regression Results}
\label{tab:ridge_results}
\begin{tabular}{lc}
\toprule
Variable & Coefficient \\
\midrule
SameCountry & -0.001 \\
model_gemini-1.5-flash & 0.000 \\
model_gpt-3.5-turbo & 0.027 \\
model_meta-llama-3-70b-instruct & 0.034 \\
model_mixtral-8x22b-instruct & 0.023 \\
model_qwen1.5-32b-chat & -0.012 \\
\bottomrule
\end{tabular}
\end{table}
"""



lasso_results = pd.DataFrame({
    'Variable': X_train.columns,
    'Coefficient': lasso.coef_
})

# Filter out variables with zero coefficients (optional, since Lasso sets some to zero)
lasso_results = lasso_results[lasso_results['Coefficient'] != 0]
