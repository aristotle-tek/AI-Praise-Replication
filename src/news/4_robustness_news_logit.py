# For robustness, we re-run the analysis with a different measure of ideology
# Data from allsides - AllSides Media Bias Ratings by AllSides.com are licensed under a Creative Commons Attribution-NonCommercial 4.0 International License. You may use this data for research or noncommercial purposes provided you include this attribution.
# https://www.kaggle.com/datasets/supratimhaldar/allsides-ratings-of-bias-in-electronic-media/data

import os
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

# dict to map from Ad Fontes to AllSides
adf_2allsides = {'ABC': 'ABC News (Online)',
 'AFP': 'NA',
 'AP': 'National Geographic',
 'Al Jazeera US/Canada News': 'Al Jazeera',
 'Alternet': 'AlterNet',
 'Axios': 'Axios',
 'BBC': 'BBC News',
 'Bipartisan Report': 'NA',
 'Bloomberg': 'Bloomberg',
 'Breitbart': 'Breitbart News',
 'Business Insider': 'NA',
 'BuzzFeed News': 'BuzzFeed News',
 'CBS': 'CBS News (Online)',
 'CNN': 'CNN (Online News)',
 'CSPAN': 'C-SPAN',
 'Christian Science Monitor': 'Christian Science Monitor',
 'Conservative Tribune': 'NA',
 'Daily Beast': 'Daily Beast',
 'Daily Caller': 'The Daily Caller',
 'Daily Kos': 'Daily Kos',
 'Daily Mail': 'Daily Mail',
 'Daily Signal': 'The Daily Signal',
 'Daily Wire': 'The Daily Wire',
 'David Wolfe': 'NA',
 'Democracy Now': 'Democracy Now',
 'Drudge Report': 'Drudge Report',
 'Financial Times': 'Financial Times',
 'Fiscal Times': 'Fiscal Times',
 'Forbes': 'Forbes',
 'Foreign Policy': 'Foreign Policy',
 'Fortune': 'Fortune',
 'Forward Progressives': 'NA',
 'Fox': 'Fox Business',
 'FreeSpeech TV': 'NA',
 'Guacamoley': 'NA',
 'Huffington Post': 'HuffPost',
 'IJR': 'NA',
 'InfoWars': 'InfoWars',
 'Intercept': 'The Intercept',
 'Jacobin': 'Jacobin',
 'LA Times': 'Los Angeles Times',
 'MSNBC': 'MSNBC',
 'Marketwatch': 'MarketWatch',
 'Mic': 'MichelleMalkin.com',
 'Mother Jones': 'Mother Jones',
 'NBC': 'CNBC',
 'NPR': 'NPR (Online News)',
 'National Enquirer': 'NA',
 'National Review': 'National Review',
 'New Republic': 'New Republic',
 'New York Post': 'New York Post (News)',
 'New York Times': 'New York Times (News)',
 'News and Guts': 'NA',
 'NewsMax': 'Newsmax (News)',
 'OAN': 'One America News Network (OAN)',
 'OZY': 'NA',
 'Occupy Democrats': 'NA',
 'PBS': 'PBS NewsHour',
 'PJ Media': 'PJ Media',
 'Palmer Report': 'NA',
 'Patribotics': 'NA',
 'Politico': 'Politico',
 'ProPublica': 'ProPublica',
 'Quartz': 'Quartz',
 'Reason': 'Reason',
 'RedState': 'RedState',
 'Reuters': 'Reuters',
 'Second Nexus': 'NA',
 'ShareBlue': 'NA',
 'Slate': 'Slate',
 'Talking Points Memo': 'NA',
 'The Advocate': 'The Advocate',
 'The American Conservative': 'The American Conservative',
 'The Atlantic': 'The Atlantic',
 'The Blaze': 'TheBlaze.com',
 'The Economist': 'The Economist',
 'The Federalist': 'The Federalist',
 'The Gateway Pundit': 'The Gateway Pundit',
 'The Guardian': 'The Guardian',
 'The Hill': 'The Hill',
 'The Nation': 'The Nation',
 'The New Yorker': 'The New Yorker',
 'The Skimm': 'NA',
 'The Week': 'The Week - News',
 'The Weekly Standard': 'The Weekly Standard',
 'The Young Turks': 'NA',
 'Think Progress': 'NA',
 'Time': 'Chicago Sun-Times',
 'Truthout': 'TruthOut',
 'Twitchy': 'NA',
 'USA Today': 'USA TODAY',
 'Vanity Fair': 'Vanity Fair',
 'Vice': 'Vice',
 'Vox': 'Vox',
 'WND': 'WND.com',
 'Wall Street Journal': 'Wall Street Journal (News)',
 'Washington Examiner': 'Washington Examiner',
 'Washington Free Beacon': 'Washington Free Beacon',
 'Washington Monthly': 'Washington Monthly',
 'Washington Post': 'Washington Post',
 'Washington Times': 'Washington Times',
 'Wonkette': 'NA',
 'WorldTruth.Tv': 'NA'}


#--------------------------------------
# Merge
#--------------------------------------

dfa = pd.read_csv(data_folder / "news" / "news_praise_scores_all.csv")

dfa = dfa[pd.notnull(dfa.transformed_score)]
dfa.index=range(len(dfa))


alls = pd.read_csv(data_folder /"news" / "allsides.csv")

set(alls.name)


dfa.rename(columns={'ideology': 'ideology', 'vertical': 'trustworthiness', 'correctedcode': 'praise_index'}, inplace=True)


dfa['allsides'] = dfa['name'].map(adf_2allsides)

dfa = dfa[dfa.allsides != 'NA']

alls['allsides'] = alls['name']

dfa = pd.merge(dfa, alls[['allsides','bias']] , on='allsides')

# code labels to ordinal
ideol_dict = {
    "left": -2,
    "left-center": -1,
    "center": 0,
    "allsides": 0, # `AllSides Balance Certification represents a higher standard that is more difficult to achieve.
    "right-center": 1,
    "right": 2,
}

dfa['allsides_ideology'] = dfa['bias'].map(ideol_dict)

dfa.allsides_ideology.value_counts()


#--------------------------------------
# Summary stats 
#--------------------------------------


summary_stats = dfa[['praise_index', 'ideology', 'allsides_ideology', 'trustworthiness']].agg(['mean', 'median', 'std', 'min', 'max'])


latex_summary_stats = summary_stats.to_latex(
    index=True,
    column_format='lccc',
    float_format="%.3f",
    caption="Summary Statistics for Praise Index, Ideology, and Trustworthiness",
    label="tab:newsstats"
)

print(latex_summary_stats)

with open(tables_folder + 'news_summary_stats.tex', 'w') as f:
    f.write(latex_summary_stats)



#--------------------------------------
# correlations
#--------------------------------------

dfa['AS_Ideology_Squared'] = dfa['allsides_ideology'] ** 2

dfa['Ideology_Squared'] = dfa['ideology'] ** 2

dfa.rename(columns={'ideology': 'Ideology', 'trustworthiness': 'Trustworthiness', 'praise_index': 'Praise_Index'}, inplace=True)


correlations_pearson = dfa[['Praise_Index', 'Ideology', 'allsides_ideology', 'Trustworthiness', 'Ideology_Squared', 'AS_Ideology_Squared']].corr(method='pearson')


correlations_pearson.rename(columns={'Praise_Index': 'Praise Index', 
                                     'Ideology': 'Ideology', 
                                     'allsides_ideology': 'AllSides Ideology',
                                     'Trustworthiness': 'Trustworthiness', 
                                     'Ideology_Squared': 'Ideology Squared', 
                                     'AS_Ideology_Squared': 'Ideology Squared (AllSides)'}, 
                            index={'Praise_Index': 'Praise Index', 
                                   'Ideology': 'Ideology', 
                                   'allsides_ideology': 'AllSides Ideology',
                                   'Trustworthiness': 'Trustworthiness', 
                                    'Ideology_Squared': 'Ideology Squared', 
                                   'AS_Ideology_Squared': 'Ideology Squared (AllSides)'}, 
                            inplace=True)

latex_pearson_corr = correlations_pearson.to_latex(
    index=True,
    column_format='lcccc',
    float_format="%.3f",
    caption="Pearson Correlations between Praise Index, Ideology, Ideology Squared, and Trustworthiness",
    label="tab:newscorrs"
)

print(correlations_pearson)

with open(tables_folder + 'news_correlations_AS.tex', 'w') as f:
    f.write(latex_pearson_corr)


"""
                             Praise Index  Ideology  ...  Ideology Squared  Ideology Squared (AllSides)
Praise Index                     1.000000 -0.053164  ...         -0.055073                    -0.032846
Ideology                        -0.053164  1.000000  ...          0.418102                     0.034689
AllSides Ideology               -0.041996  0.808987  ...          0.433020                     0.059702
Trustworthiness                  0.056120 -0.476367  ...         -0.793558                    -0.578016
Ideology Squared                -0.055073  0.418102  ...          1.000000                     0.582005
Ideology Squared (AllSides)     -0.032846  0.034689  ...          0.582005                     1.000000
"""

print(latex_pearson_corr)


# Spearman rank corr
correlations_sp = dfa[['Praise_Index', 'Ideology', 'allsides_ideology', 'Trustworthiness', 'Ideology_Squared', 'AS_Ideology_Squared']].corr(method='spearman')

correlations_sp.rename(columns={'Praise_Index': 'Praise Index', 
                                     'Ideology': 'Ideology', 
                                     'allsides_ideology': 'AllSides Ideology',
                                     'Trustworthiness': 'Trustworthiness', 
                                     'Ideology_Squared': 'Ideology Squared', 
                                     'AS_Ideology_Squared': 'Ideology Squared (AllSides)'}, 
                            index={'Praise_Index': 'Praise Index', 
                                   'Ideology': 'Ideology', 
                                   'allsides_ideology': 'AllSides Ideology',
                                   'Trustworthiness': 'Trustworthiness', 
                                    'Ideology_Squared': 'Ideology Squared', 
                                   'AS_Ideology_Squared': 'Ideology Squared (AllSides)'}, 
                            inplace=True)

latex_sp_corr = correlations_sp.to_latex(
    index=True,
    column_format='lcccc',
    float_format="%.3f",
    caption="Spearman Correlations between Praise Index, Ideology, Ideology Squared, and Trustworthiness",
    label="tab:spearmannewscorrs"
)

print(correlations_sp)


"""


                             Praise Index  Ideology  ...  Ideology Squared  Ideology Squared (AllSides)
Praise Index                     1.000000 -0.042474  ...         -0.023327                    -0.024730
Ideology                        -0.042474  1.000000  ...          0.130542                    -0.072982
AllSides Ideology               -0.034410  0.798709  ...          0.274818                    -0.055733
Trustworthiness                  0.030650 -0.247450  ...         -0.785347                    -0.600503
Ideology Squared                -0.023327  0.130542  ...          1.000000                     0.690826
Ideology Squared (AllSides)     -0.024730 -0.072982  ...          0.690826                     1.000000

#  & Praise Index & Ideology & AllSides Ideology & Trustworthiness & Ideology Squared (AllSides) \\
 & Praise Index & Ideology & AllSides Ideology & Trustworthiness & Ideology Squared & Ideology Squared (AllSides) \\
\midrule
Praise Index & 1.000 & -0.042 & -0.034 & 0.031 & -0.023 & -0.025 \\
Ideology & -0.042 & 1.000 & 0.799 & -0.247 & 0.131 & -0.073 \\
AllSides Ideology & -0.034 & 0.799 & 1.000 & -0.353 & 0.275 & -0.056 \\
Trustworthiness & 0.031 & -0.247 & -0.353 & 1.000 & -0.785 & -0.601 \\
Ideology Squared & -0.023 & 0.131 & 0.275 & -0.785 & 1.000 & 0.691 \\
Ideology Squared (AllSides) & -0.025 & -0.073 & -0.056 & -0.601 & 0.691 & 1.000 \\
\bottomrule
\end{tabular}
\end{table}
"""


# scatterplt of Ideology vs allsides_ideology

plt.figure(figsize=(8, 6))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Source Serif Pro"], 
    "font.size": 14,   
    "axes.labelsize": 14, 
    "axes.titlesize": 16, 
    "legend.fontsize": 14,
    "xtick.labelsize": 14,  
    "ytick.labelsize": 14 
})

plt.scatter(dfa['Ideology'], dfa['allsides_ideology'], alpha=0.6) # alpha fails...
plt.title('')
plt.xlabel('Ideology (Ad Fontes)')
plt.ylabel('Ideology (AllSides categories converted to -2, -1, 0, 1, 2)')
plt.grid(True, linestyle='--', alpha=0.2)
plt.savefig(plotdir / 'scatter_ideology_measures.pdf', format='pdf')
#plt.show()


#-- look at those labeled differently -
filtered_rows = dfa[(dfa['Ideology'] < 0) & (dfa['allsides_ideology'] > 0)]

print(set(filtered_rows['name']))

#{'Daily Signal', 'Mic', 'NewsMax'}


#------------------------------
# Ordered Logit
#------------------------------


# Ensure 'correctedcode' is ordered cat
dfa['correctedcode'] = dfa['transformed_score'].astype(int).astype('category')
dfa['correctedcode'] = dfa['correctedcode'].cat.as_ordered()

# Center 'ideology' and 'vertical' 
dfa['ideology_centered'] = dfa['allsides_ideology'] - dfa['allsides_ideology'].mean()
dfa['vertical_centered'] = dfa['Trustworthiness'] - dfa['Trustworthiness'].mean()

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
    
    # for each model, create list with coefs & se
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
    
    n_str = f"${res.nobs}$"
    pseudo_r2_str = f"${res.prsquared:.3f}$"
    model_column.append(n_str)
    model_column.append(pseudo_r2_str)
    
    if i == 0:
        table_index.append("N")
        table_index.append("Pseudo R^2")
    
    table_data.append(model_column)

table_data = list(map(list, zip(*table_data)))  # transpose

latex_df = pd.DataFrame(table_data, columns=model_results.keys(), index=table_index)

latex_table = latex_df.to_latex(column_format="l" + "c" * len(latex_df.columns), escape=False)

print(latex_table)


with open(tables_folder + 'news_logits_Allsides.tex', 'w') as f:
    f.write(latex_table)



#--------------------------------------------
# Calc Average Marginal Effects (AMEs)
#--------------------------------------------


ame_results = {}

for model_name, res in model_results.items():
    df_model = dfa[dfa['model'] == model_name].copy()
    df_model.dropna(subset=['correctedcode'] + exog_vars, inplace=True)
    
    # Calc std
    std_ideology = df_model['ideology_centered'].std()
    std_vertical = df_model['vertical_centered'].std()
    
    ame_ideology = []
    ame_vertical = []
    
    for _, row in df_model.iterrows():
        # Baseline pred at observed values
        exog_base = row[exog_vars].astype(float).values.reshape(1, -1)
        base_probs = res.model.predict(res.params, exog=exog_base, which='prob')
        
        # pred prob with +1 std dev in ideology
        exog_ideology = exog_base.copy()
        exog_ideology[0, exog_vars.index('ideology_centered')] += std_ideology
        ideology_probs = res.model.predict(res.params, exog=exog_ideology, which='prob')
        
        #  trustworthiness (vertical)
        exog_vertical = exog_base.copy()
        exog_vertical[0, exog_vars.index('vertical_centered')] += std_vertical
        vertical_probs = res.model.predict(res.params, exog=exog_vertical, which='prob')
        
        # Marginal effects
        ame_ideology.append(ideology_probs - base_probs)
        ame_vertical.append(vertical_probs - base_probs)
    
    # avg marginal effects
    ame_ideology_avg = np.mean(ame_ideology, axis=0).flatten()
    ame_vertical_avg = np.mean(ame_vertical, axis=0).flatten()
    
    ame_results[model_name] = {
        'AME_Ideology': ame_ideology_avg,
        'AME_Trustworthiness': ame_vertical_avg
    }


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
    renamed_model_name = model_name_map.get(model_name, model_name)
    for category, effect in enumerate(effects['AME_Ideology']):
        # only show model name once
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
    columns=['Model', 'Outcome', 'Ideology (AllSides)', 'Trustworthiness', 'Ratio']
)


latex_table = ame_df.to_latex(
    index=False, header=True,
    column_format="l" + "c" * (ame_df.shape[1] - 1),
    caption="News: ordered logit predictive marginal effects (AllSides))",
    label="tab:news_PME_AS"
)

print(latex_table)


with open(tables_folder + 'ame_results_with_ratio_AS.tex', 'w') as f:
    f.write(latex_table)






