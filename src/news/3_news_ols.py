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



dfa = pd.read_csv(data_folder + "news_praise_scores_all.csv")
dfa = dfa[pd.notnull(dfa.transformed_score)]
dfa.index = range(len(dfa))

# Center 'ideology' and 'vertical' columns
dfa['ideology_centered'] = dfa['ideology'] - dfa['ideology'].mean()
dfa['vertical_centered'] = dfa['vertical'] - dfa['vertical'].mean()
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
# output to LaTeX

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

# Convert to LaTeX format
latex_table = latex_df.to_latex(column_format="l" + "c" * len(latex_df.columns), escape=False)
print(latex_table)


#----------------------
# plot across levels of ideology
#----------------------


ideology_range = np.linspace(dfa['ideology'].min(), dfa['ideology'].max(), 100)


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
        'ideology_centered': ideology_range - dfa['ideology'].mean(),
        'ideology_sq': (ideology_range - dfa['ideology'].mean()) ** 2,
        'vertical_centered': df_model['vertical_centered'].mean(),
        'negative_example': 0  # 0 as baseline
    })

    res = model_results[model_name]
    predictions = res.get_prediction(pred_data)
    pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI

    plt.plot(
        ideology_range, 
        pred_summary['mean'], 
        label=model_name_map.get(model_name, model_name), # nicer model names
        color=colors[idx]
    )
    plt.fill_between(
        ideology_range, 
        pred_summary['mean_ci_lower'], 
        pred_summary['mean_ci_upper'], 
        color=colors[idx], 
        alpha=0.1
    )


plt.xlabel("Ideology")
plt.ylabel("Predicted Praise Score")
plt.title("")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlim(dfa['ideology'].min(), dfa['ideology'].max())
plt.tight_layout()

plt.savefig(plotdir / 'news_ols_ideol.pdf', format='pdf')
# use e.g. pdftops -eps /path/to/news_ols_ideol.pdf /path/to/news_ols_ideol.eps


#plt.show()



#------------------------------
# Plot residualized
#------------------------------



dfa = pd.read_csv(data_folder / "news_praise_scores_all.csv")

dfa = dfa[pd.notnull(dfa.transformed_score)]

dfa.index=range(len(dfa))

residformula = 'correctedcode ~ vertical -1' 
residmodel = smf.ols(residformula, data=dfa).fit()


dfa['residuals'] = residmodel.resid


average_scores = dfa.groupby(['name', 'ideology']).agg({
    'correctedcode': 'mean',
    'residuals': 'mean'
}).reset_index()



plt.figure(figsize=(14, 8))
plt.scatter(average_scores['ideology'], average_scores['correctedcode'], label='Observed Praise Index', color='blue')
plt.scatter(average_scores['ideology'], average_scores['residuals'], label='Residualized Scores', color='red')

# Adding labels
for i in range(len(average_scores)):
    plt.text(average_scores['ideology'][i], average_scores['correctedcode'][i], average_scores['name'][i], fontsize=8, ha='right')
    plt.text(average_scores['ideology'][i], average_scores['residuals'][i], average_scores['name'][i], fontsize=8, ha='left')

plt.xlabel('Ideology Score')
plt.ylabel('Score')
plt.title('Observed Praise Index and Residualized Scores by Ideology')
plt.legend()
plt.tight_layout()
plt.savefig(plotdir + 'news_resid.eps', format='eps')
plt.savefig(plotdir + 'news_resid.pdf', format='pdf')
#plt.show()








#---- version 2 - reduce overplot


import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

def overlap_fraction(bbox1, bbox2):
    left = max(bbox1.x0, bbox2.x0)
    right = min(bbox1.x1, bbox2.x1)
    bottom = max(bbox1.y0, bbox2.y0)
    top = min(bbox1.y1, bbox2.y1)
    if right <= left or top <= bottom:
        return 0.0
    intersect_area = (right - left) * (top - bottom)
    area1 = (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0)
    area2 = (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0)
    return intersect_area / min(area1, area2)


overlap_tolerance=0.2
x = average_scores['ideology']
y_obs = average_scores['correctedcode']
y_res = average_scores['residuals']


plt.figure(figsize=(14, 8))
plt.scatter(x, y_obs, label='Observed Praise Index', color='blue', s=20, alpha=0.3)
plt.scatter(x, y_res, label='Residualized Scores', color='red', s=20, alpha=0.3)

used_bboxes = []

for i in range(len(average_scores)):
    label_text_obs = plt.text(
        x[i], y_obs[i], average_scores['name'][i],
        fontsize=10, ha='right'
    )
    label_text_res = plt.text(
        x[i], y_res[i], average_scores['name'][i],
        fontsize=10, ha='left'
    )

    renderer = plt.gca().get_figure().canvas.get_renderer()
    bbox_obs = label_text_obs.get_window_extent(renderer=renderer)
    bbox_res = label_text_res.get_window_extent(renderer=renderer)

    # check observed label
    for existing_bbox in used_bboxes:
        if overlap_fraction(bbox_obs, existing_bbox) > overlap_tolerance:
            label_text_obs.remove()
            break
    else:
        used_bboxes.append(bbox_obs)

    # check resid label
    for existing_bbox in used_bboxes:
        if overlap_fraction(bbox_res, existing_bbox) > overlap_tolerance:
            label_text_res.remove()
            break
    else:
        used_bboxes.append(bbox_res)

plt.xlabel('Ideology Score')
plt.ylabel('Score')
plt.title('')
plt.legend()
plt.tight_layout()
plt.savefig(plotdir / 'news_resid_min_labels.pdf', format='pdf')
# plt.show()

