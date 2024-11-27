# news_regr.py



import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.iolib.summary2 import summary_col


import matplotlib.pyplot as plt
import seaborn as sns


from pathlib import Path


base_path = Path(__file__).resolve().parent.parent

import itertools


code_path = base_path / "src"

plotdir = base_path / "plots"

tables_folder = base_path / "tables"
data_folder = base_path / "data" / "news"

if not os.path.exists(plotdir):
    os.makedirs(plotdir)

if not os.path.exists(tables_folder):
    os.makedirs(tables_folder)


#--------------------------------------
# Summary stats 
#--------------------------------------

dfa = pd.read_csv(data_folder + "news_praise_scores_all.csv")

dfa = dfa[pd.notnull(dfa.transformed_score)]
dfa.index=range(len(dfa))


# Rename cols
dfa.rename(columns={'ideology': 'ideology', 'vertical': 'trustworthiness', 'correctedcode': 'praise_index'}, inplace=True)


# 1. Summary stats: means, medians, st dev.
summary_stats = dfa[['praise_index', 'ideology', 'trustworthiness']].agg(['mean', 'median', 'std', 'min', 'max'])


latex_summary_stats = summary_stats.to_latex(
    index=True,
    column_format='lccc',
    float_format="%.3f",
    caption="Summary Statistics for Praise Index, Ideology, and Trustworthiness",
    label="tab:newsstats"
)


with open(tables_folder + 'news_summary_stats.tex', 'w') as f:
    f.write(latex_summary_stats)





# Count of extreme news sources 
news = dfa.drop_duplicates('name')
ideol_mn = news['ideology'].mean()
ideol_std = news['ideology'].std()


one_std_above = news[news['ideology'] > (ideol_mn + ideol_std) ].shape[0]
one_std_below = news[news['ideology'] < (ideol_mn - ideol_std)].shape[0]

two_std_above = news[news['ideology'] > (ideol_mn + (2 * ideol_std))].shape[0]
two_std_below = news[news['ideology'] < (ideol_mn - (2 * ideol_std))].shape[0]

one_std_above, one_std_below, two_std_above, two_std_below

# (23, 19, 1, 0)

# news.name[news['ideology'] < (ideol_mn - (1 * ideol_std))] # left
# news.name[news['ideology'] > (ideol_mn + (1 * ideol_std))] # right





#--------------------------------------
# correlations
#--------------------------------------


dfa = pd.read_csv(data_folder + "news_praise_scores_all.csv")


dfa = dfa[pd.notnull(dfa.transformed_score)]
dfa.index = range(len(dfa))

dfa.rename(columns={'ideology': 'Ideology', 'vertical': 'Trustworthiness', 'correctedcode': 'Praise_Index'}, inplace=True)


dfa['Ideology_Squared'] = dfa['Ideology'] ** 2


correlations_pearson = dfa[['Praise_Index', 'Ideology', 'Trustworthiness', 'Ideology_Squared']].corr(method='pearson')


correlations_pearson.rename(columns={'Praise_Index': 'Praise Index', 
                                     'Ideology': 'Ideology', 
                                     'Trustworthiness': 'Trustworthiness', 
                                     'Ideology_Squared': 'Ideology Squared'}, 
                            index={'Praise_Index': 'Praise Index', 
                                   'Ideology': 'Ideology', 
                                   'Trustworthiness': 'Trustworthiness', 
                                   'Ideology_Squared': 'Ideology Squared'}, 
                            inplace=True)

latex_pearson_corr = correlations_pearson.to_latex(
    index=True,
    column_format='lcccc',
    float_format="%.3f",
    caption="Pearson Correlations between Praise Index, Ideology, Ideology Squared, and Trustworthiness",
    label="tab:newscorrs"
)


with open(tables_folder + 'news_correlations.tex', 'w') as f:
    f.write(latex_pearson_corr)

correlations_pearson



#------------------------------
# Ordered Logit
#------------------------------

dfa = pd.read_csv(data_folder + "news_praise_scores_all.csv")
dfa = dfa[pd.notnull(dfa.transformed_score)]
dfa.index = range(len(dfa))

# Ensure 'correctedcode' is ordered cat
dfa['correctedcode'] = dfa['transformed_score'].astype(int).astype('category')
dfa['correctedcode'] = dfa['correctedcode'].cat.as_ordered()

# Center 'ideology' and 'vertical' 
dfa['ideology_centered'] = dfa['ideology'] - dfa['ideology'].mean()
dfa['vertical_centered'] = dfa['vertical'] - dfa['vertical'].mean()

dfa['ideology_sq'] = dfa['ideology_centered'] **2

dfa['negative_example'] = dfa['multiplier'].apply(lambda x: 1 if x == -1 else 0)


unique_models = dfa['model'].unique()

exog_vars = ['ideology_centered', 'ideology_sq', 'vertical_centered', 'negative_example']


model_results = {}

for model_name in unique_models:
    df_model = dfa[dfa['model'] == model_name].copy()

    df_model.dropna(subset=['correctedcode'] + exog_vars, inplace=True)
    
    exog = df_model[exog_vars].astype(float)
    
    model = OrderedModel(df_model['correctedcode'], exog, distr='logit')
    res = model.fit(method='bfgs', maxiter=1000)
    
    model_results[model_name] = res
    print(f"Results for model {model_name}:")
    print(res.summary())


# for latex out
table_data = []
table_index = []

for i, (model_name, res) in enumerate(model_results.items()):
    coefs = res.params
    std_errs = res.bse
    pvalues = res.pvalues
    
    # For each model, create list with coefs & se
    model_column = []
    for param, coef in coefs.items():
        stars = ""
        if pvalues[param] < 0.01:
            stars = "^{***}"
        elif pvalues[param] < 0.05:
            stars = "^{**}"
        elif pvalues[param] < 0.1:
            stars = "^{*}"
        
        # fmt stars for latex
        coef_str = f"${coef:.3f}{stars}$"
        std_err_str = f"$({std_errs[param]:.3f})$"
        
        model_column.append(coef_str)
        model_column.append(std_err_str)
        
        if i == 0:
            table_index.append(f"{param} coef")
            table_index.append(f"{param} SE")
    
    # Add N and r2 for the model
    n_str = f"${res.nobs}$"
    pseudo_r2_str = f"${res.prsquared:.3f}$"
    model_column.append(n_str)
    model_column.append(pseudo_r2_str)
    
    if i == 0:
        table_index.append("N")
        table_index.append("Pseudo R^2")
    
    table_data.append(model_column)

table_data = list(map(list, zip(*table_data)))  # Transpose


latex_df = pd.DataFrame(table_data, columns=model_results.keys(), index=table_index)

latex_table = latex_df.to_latex(column_format="l" + "c" * len(latex_df.columns), escape=False)

print(latex_table)


with open(tables_folder + 'news_logits.tex', 'w') as f:
    f.write(latex_table)



#--------------------------------------------
# Calc Average Marginal Effects (AMEs)
#--------------------------------------------

# to output names of models:
model_name_map = {
    "qwen": "Qwen-1.5-32b",
    "gpt35": "GPT-3.5-turbo",
    "llama3": "Llama-3 70B",
    "gemini": "Gemini-1.5-flash",
    "mixtral": "Mixtral-8x22b",
    "claude": "Claude-3-sonnet"
}



ame_data = []
table_index = []

for model_name, effects in ame_results.items():
    renamed_model_name = model_name_map.get(model_name, model_name)  # Map model name
    for category, effect in enumerate(effects['AME_Ideology']):
        # Only show model name once
        model_display_name = renamed_model_name if category == 0 else ""
        
        # rename outcome to -1, 0, 1
        outcome_category = category - 1
        
        ideology_str = f"{effect:.3f}"
        trustworthiness_str = f"{effects['AME_Trustworthiness'][category]:.3f}"
        
        # calc ratio
        trustworthiness_effect = effects['AME_Trustworthiness'][category]
        ratio = abs(trustworthiness_effect / effect  ) if effect != 0 else "N/A"
        ratio_str = f"{ratio:.3f}" if ratio != "N/A" else ratio
        
        ame_data.append([model_display_name, outcome_category, ideology_str, trustworthiness_str, ratio_str])

ame_df = pd.DataFrame(
    ame_data, 
    columns=['Model', 'Outcome', 'Ideology', 'Trustworthiness', 'Ratio']
)


latex_table = ame_df.to_latex(
    index=False, header=True,
    column_format="l" + "c" * (ame_df.shape[1] - 1),
    caption="News: ordered logit predictive marginal effects",
    label="tab:news_PME"
)

print(latex_table)


with open(tables_folder + 'ame_results_with_ratio.tex', 'w') as f:
    f.write(latex_table)






