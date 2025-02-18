#3_news_ols.py


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from pathlib import Path

code_path = base_path / "src"



plotdir = base_path / "plots"

tables_folder = base_path / "tables"
data_folder = base_path / "data" / "news"

if not os.path.exists(plotdir):
    os.makedirs(plotdir)

if not os.path.exists(tables_folder):
    os.makedirs(tables_folder)



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


dfa['allsides'] = dfa['name'].map(adf_2allsides)

dfa = dfa[dfa.allsides != 'NA']

alls['allsides'] = alls['name']

dfa = pd.merge(dfa, alls[['allsides','bias']] , on='allsides')

# Now map from categories to approx location

ideol_dict = {
    "left": -2,
    "left-center": -1,
    "center": 0,
    "allsides": 0, # AllSides Balance Certification represents a higher standard that is more difficult to achieve.
    "right-center": 1,
    "right": 2,
}

dfa['allsides_ideology'] = dfa['bias'].map(ideol_dict)


dfa.rename(columns={ 'vertical': 'trustworthiness', 'correctedcode': 'praise_index'}, inplace=True)


# center 'ideology' and 'vertical' cols
dfa['ideology_centered'] = dfa['allsides_ideology'] - dfa['allsides_ideology'].mean()
dfa['vertical_centered'] = dfa['trustworthiness'] - dfa['trustworthiness'].mean()
dfa['ideology_sq'] = dfa['ideology_centered'] ** 2

dfa['negative_example'] = dfa['multiplier'].apply(lambda x: 1 if x == -1 else 0)

unique_models = dfa['model'].unique()

exog_vars = ['ideology_centered', 'ideology_sq', 'vertical_centered', 'negative_example']

model_results = {}

for model_name in unique_models:
    df_model = dfa[dfa['model'] == model_name].copy()
    df_model.dropna(subset=['transformed_score'] + exog_vars, inplace=True)
    
    exog = df_model[exog_vars].astype(float)
    exog = sm.add_constant(exog)
    
    endog = df_model['transformed_score']  # continuous
    
    # OLS + cluster-robust se
    ols_model = sm.OLS(endog, exog)
    res = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df_model['name']})
    
    model_results[model_name] = res
    print(f"Results for model {model_name}:")
    print(res.summary())




#------------
# LaTeX

table_data = []
table_index = []

for i, (model_name, res) in enumerate(model_results.items()):
    coefs = res.params
    std_errs = res.bse
    pvalues = res.pvalues
    
    model_column = []
    for param, coef in coefs.items():
        stars = ""
        if pvalues[param] < 0.01:
            stars = "^{***}"
        elif pvalues[param] < 0.05:
            stars = "^{**}"
        elif pvalues[param] < 0.1:
            stars = "^{*}"
        
        coef_str = f"${coef:.3f}{stars}$"
        std_err_str = f"$({std_errs[param]:.3f})$"
        
        model_column.append(coef_str)
        model_column.append(std_err_str)
        
        if i == 0:
            table_index.append(f"{param} coef")
            table_index.append(f"{param} SE")
    
    n_str = f"${int(res.nobs)}$"
    r2_str = f"${res.rsquared:.3f}$"
    model_column.append(n_str)
    model_column.append(r2_str)
    
    if i == 0:
        table_index.append("N")
        table_index.append("R-squared")
    
    table_data.append(model_column)


table_data = list(map(list, zip(*table_data)))
latex_df = pd.DataFrame(table_data, columns=model_results.keys(), index=table_index)

latex_table = latex_df.to_latex(column_format="l" + "c" * len(latex_df.columns), escape=False)
print(latex_table)



"""
\begin{tabular}{lcccccc}
\toprule
 & claude & gpt35 & gemini & mixtral & llama3 & qwen \\
\midrule
const coef & $0.111^{***}$ & $0.857^{***}$ & $0.389^{***}$ & $0.659^{***}$ & $0.696^{***}$ & $0.545^{***}$ \\
& $(0.029)$ & $(0.024)$ & $(0.040)$ & $(0.030)$ & $(0.024)$ & $(0.032)$ \\
ideology_centered coef & $0.000$ & $-0.013$ & $-0.030^{**}$ & $-0.006$ & $-0.033^{***}$ & $-0.051^{***}$ \\
 & $(0.012)$ & $(0.010)$ & $(0.015)$ & $(0.011)$ & $(0.010)$ & $(0.017)$ \\
ideology_sq coef & $-0.019$ & $-0.009$ & $-0.008$ & $-0.023^{**}$ & $-0.022^{**}$ & $-0.011$ \\
 & $(0.012)$ & $(0.009)$ & $(0.013)$ & $(0.011)$ & $(0.009)$ & $(0.016)$ \\
vertical_centered coef & $0.004^{***}$ & $0.002^{**}$ & $0.007^{***}$ & $0.003^{**}$ & $0.002$ & $0.007^{***}$ \\
 & $(0.001)$ & $(0.001)$ & $(0.002)$ & $(0.001)$ & $(0.001)$ & $(0.002)$ \\
negative_example coef & $-0.045$ & $-1.657^{***}$ & $-0.820^{***}$ & $-1.200^{***}$ & $-1.402^{***}$ & $-1.053^{***}$ \\
 & $(0.029)$ & $(0.025)$ & $(0.049)$ & $(0.032)$ & $(0.023)$ & $(0.039)$ \\
N & $1200$ & $1200$ & $1200$ & $1200$ & $1200$ & $1199$ \\
R-squared & $0.023$ & $0.780$ & $0.250$ & $0.510$ & $0.643$ & $0.421$ \\
\bottomrule
\end{tabular}

"""


#----------------------
# plot across levels of ideology
#----------------------


ideology_range = np.linspace(dfa['allsides_ideology'].min(), dfa['allsides_ideology'].max(), 100)


model_name_map = {
    "qwen": "Qwen-1.5-32b",
    "gpt35": "GPT-3.5-turbo",
    "llama3": "Llama-3 70B",
    "gemini": "Gemini-1.5-flash",
    "mixtral": "Mixtral-8x22b",
    "claude": "Claude-3-sonnet"
}


# LaTeX font
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


plt.figure(figsize=(12, 8))
plt.ylim((0.0, 1.0))
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_models)))


for idx, model_name in enumerate(unique_models):
    df_model = dfa[dfa['model'] == model_name].copy()

    pred_data = pd.DataFrame({
        'const': 1,
        'ideology_centered': ideology_range - dfa['allsides_ideology'].mean(),
        'ideology_sq': (ideology_range - dfa['allsides_ideology'].mean()) ** 2,
        'vertical_centered': df_model['vertical_centered'].mean(), 
        'negative_example': 0  # baseline 0
    })

    res = model_results[model_name]
    predictions = res.get_prediction(pred_data)
    pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI

    plt.plot(
        ideology_range, 
        pred_summary['mean'], 
        label=model_name_map.get(model_name, model_name),  # model_name, 
        color=colors[idx]
    )
    plt.fill_between(
        ideology_range, 
        pred_summary['mean_ci_lower'], 
        pred_summary['mean_ci_upper'], 
        color=colors[idx], 
        alpha=0.1
    )


plt.xlabel("Ideology (AllSides)")
plt.ylabel("Predicted Praise Score")
plt.title("")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlim(dfa['allsides_ideology'].min(), dfa['allsides_ideology'].max())
plt.tight_layout()

plt.savefig(plotdir / 'news_ols_ideol_AS.pdf', format='pdf')
#  use e.g. pdftops -eps /path/to/news_ols_ideol.pdf /path/to/news_ols_ideol.eps


#plt.show()

