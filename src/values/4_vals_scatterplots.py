

import re
import json
import pandas as pd
import numpy as np
import time
import os

import logging


import itertools
from scipy.stats import pearsonr, spearmanr, norm # earlier versions: from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats

import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from pathlib import Path


base_path = Path(__file__).resolve().parent.parent

code_path = base_path / "src"



plotdir = base_path / "data" / "values" / "plots"


coded_dir = base_path / "data" / "values" / "coded"

model_list = ['gpt-3.5-turbo', 'claude-3-sonnet-20240229', 'gemini-1.5-flash', \
'qwen1.5-32b-chat','meta-llama-3-70b-instruct',"mixtral-8x22b-instruct"]

name_map = {
    'qwen1.5-32b-chat': "Qwen-1.5-32b",
    'gpt-3.5-turbo': "GPT-3.5-turbo",
    'meta-llama-3-70b-instruct': "Llama-3 70B",
    'gemini-1.5-flash': "Gemini-1.5-flash",
    "mixtral-8x22b-instruct": "Mixtral-8x22b",
    'claude-3-sonnet-20240229': "Claude-3-Sonnet"
}

#-------------------------------------
# summary stats
#-------------------------------------

summary_rows = []

first_model = model_list[0]
first_jdf = pd.read_csv(coded_dir / f"Coded_outputs_{first_model}_sents_wcodes.csv")
desc_diff = first_jdf['Score'].describe()


summary_rows.append({
    'Model': '',  # 
    'Variable': 'Human Rating',
    'N': int(desc_diff['count']),
    'Mean': desc_diff['mean'],
    'Std': desc_diff['std'],
    'Min': desc_diff['min'],
    '25%': desc_diff['25%'],
    '50%': desc_diff['50%'],
    '75%': desc_diff['75%'],
    'Max': desc_diff['max'],
})


for whichmodel in model_list:
    renamed_model = name_map.get(whichmodel, whichmodel)
    jdf = pd.read_csv(coded_dir / f"Coded_outputs_{whichmodel}_sents_wcodes.csv")
    desc_score = jdf['difference'].describe()
    
    summary_rows.append({
        'Model': renamed_model,
        'Variable': 'Praise Score',
        'N': int(desc_score['count']),
        'Mean': desc_score['mean'],
        'Std': desc_score['std'],
        'Min': desc_score['min'],
        '25%': desc_score['25%'],
        '50%': desc_score['50%'],
        '75%': desc_score['75%'],
        'Max': desc_score['max'],
    })


summary_df = pd.DataFrame(summary_rows)

latex_table = summary_df.to_latex(
    index=False,
    float_format="%.3f",
    columns=["Model","Variable","N","Mean","Std","Min","25th","50th","75th","Max"],
    column_format="l l r r r r r r r r"  
)

print(latex_table)


#-------------------------------------
# plot
#-------------------------------------


def plot_stdized_wlabels(m2, ax, title, alpha=1.0):
    scaler = StandardScaler()
    m2[['diffscore', 'humanscore']] = scaler.fit_transform(m2[['difference', 'Score']])

    x = m2.diffscore
    y = m2.humanscore

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept

    n = len(x)
    mean_x = np.mean(x)
    t = stats.t.ppf(1 - 0.025, n - 2)
    s_err = np.sqrt(np.sum((y - line) ** 2) / (n - 2))

    clip_on = True

    for i in range(len(m2)):
        ax.annotate(m2.actionmod[i], (m2.diffscore[i], m2.humanscore[i]),
                     textcoords="offset points", xytext=(5, 5), ha='center',
                     clip_on=clip_on, alpha=alpha)  # poss transparency for annotations (alpha<1)

    ax.plot(m2.diffscore, line, color='red', label='Line of best fit')
    ax.set_title(title)
    ax.set_xlabel('', alpha=alpha) # 'Praise - difference score'
    ax.set_ylabel('', alpha=alpha) # Human rating

    ax.plot([-3, 3], [-3, 3], color='blue', linestyle='--', label='x = y')

    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-2.5, 2.5)
    ax.grid(False)



# LaTeX font and transparency
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


#------------------------------
# Correlations table
#------------------------------

correlation_rows = []

for whichmodel in model_list:
    renamed_model = name_map.get(whichmodel, whichmodel)
    jdf = pd.read_csv(coded_dir / f"Coded_outputs_{whichmodel}_sents_wcodes.csv")
    avg_diff = jdf.groupby('actionmod')['difference'].mean().reset_index()
    jdfuniqaction = jdf[['actionmod', 'Score']].drop_duplicates()
    m2 = pd.merge(avg_diff, jdfuniqaction, on='actionmod', how='left')

    pearson_corr, pearson_pval = pearsonr(m2.difference, m2.Score)
    spearman_corr = m2[['difference', 'Score']].corr(method='spearman').iloc[0, 1]

    correlation_rows.append({
        'Model': renamed_model,
        'Pearson r': f"{pearson_corr:.3f}",
        'p-value': f"{pearson_pval:.3f}",
        'Spearman $\rho$': f"{spearman_corr:.3f}",
    })

correlation_df = pd.DataFrame(correlation_rows)

latex_table = correlation_df.to_latex(
    index=False,
    float_format="%.3f",
    column_format="l r r r",
    caption="Pearson and Spearman Correlations between Praise Scores and Human Ratings",
    label="tab:correlations"
)

print(latex_table)

#------------------------------
# Scatterplot with line of best fit
#------------------------------ 

# NB: The final version uses the code below to reduce overplotting

num_models = len(model_list)
num_rows = (num_models + 1) // 2
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
axes = axes.flatten()

standardize = True
include_corrs = False

for idx, whichmodel in enumerate(model_list):
    print("Model: ", whichmodel)

    jdf = pd.read_csv(coded_dir + f"Coded_outputs_{whichmodel}_sents_wcodes.csv")

    average_difference_by_action = jdf.groupby('actionmod')['difference'].mean().reset_index()
    jdfuniqaction = jdf[['actionmod', 'Score']].drop_duplicates()
    m2 = pd.merge(average_difference_by_action, jdfuniqaction, on='actionmod', how='left')
    if standardize:
        scaler = StandardScaler()
        m2[['difference', 'Score']] = scaler.fit_transform(m2[['difference', 'Score']])
    corr = np.corrcoef(m2.difference, m2.Score)
    pr = pearsonr(m2.difference, m2.Score)
    spearmancorr = m2[['difference','Score']].corr(method='spearman')
    print("Spearman corr: ", spearmancorr.values[0,1])
    if include_corrs:
        title = f'{whichmodel.capitalize()}, Spearman: {spearmancorr.iloc[0,1]:.3f}\nPearson: {corr[0, 1]:.3f}, p-val: {pr[1]:.3f}'
    else:
        title = f'Model: {whichmodel}'
    plot_stdized_wlabels(m2, axes[idx], title, alpha=0.7)

# Hide unused subplots if num_models is odd
for i in range(num_models, num_rows * 2):
    fig.delaxes(axes[i])

plt.tight_layout()
if include_corrs:
    plt.savefig(plotdir / "vals_scatter_all_labels_wcorrs.pdf")
else:
    plt.savefig(plotdir / "vals_scatter_all_labels.pdf")


#-----------
# version 2 - reduce overplot



def overlap_fraction(bbox1, bbox2):
    left = max(bbox1.x0, bbox2.x0)
    right = min(bbox1.x1, bbox2.x1)
    bottom = max(bbox1.y0, bbox2.y0)
    top = min(bbox1.y1, bbox2.y1)
    intersection_width = right - left
    intersection_height = top - bottom
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0
    intersect_area = intersection_width * intersection_height
    area1 = (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0)
    area2 = (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0)
    smaller_area = min(area1, area2)
    return intersect_area / smaller_area

def plot_stdized_wlabels(m2, ax, title, alpha=1.0, overlap_tolerance=0.3):
    scaler = StandardScaler()
    m2[['diffscore', 'humanscore']] = scaler.fit_transform(m2[['difference', 'Score']])
    x = m2.diffscore
    y = m2.humanscore
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept

    ax.plot(x, line, color='red', label='Line of best fit')
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.plot([-3, 3], [-3, 3], color='blue', linestyle='--', label='x = y')
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-2.5, 2.5)
    ax.grid(False)

    used_bboxes = []
    for i in range(len(m2)):
        text_obj = ax.annotate(
            m2.actionmod[i],
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(5, 5),
            ha='center',
            clip_on=True,
            alpha=alpha
        )
        ax.figure.canvas.draw()
        bbox = text_obj.get_window_extent(
            renderer=ax.figure.canvas.get_renderer()
        )
        too_much_overlap = False
        for existing_bbox in used_bboxes:
            if overlap_fraction(bbox, existing_bbox) > overlap_tolerance:
                too_much_overlap = True
                break
        if not too_much_overlap:
            used_bboxes.append(bbox)
        else:
            text_obj.remove()




num_models = len(model_list)
num_rows = (num_models + 1) // 2
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
axes = axes.flatten()

standardize = True
include_corrs = False # (moved to a table)



for idx, whichmodel in enumerate(model_list):
    renamed_model_name = name_map.get(whichmodel, whichmodel)
    print(renamed_model_name)
    jdf = pd.read_csv(coded_dir / f"Coded_outputs_{whichmodel}_sents_wcodes.csv")

    avg_diff = jdf.groupby('actionmod')['difference'].mean().reset_index()
    jdfuniqaction = jdf[['actionmod', 'Score']].drop_duplicates()
    m2 = pd.merge(avg_diff, jdfuniqaction, on='actionmod', how='left')

    if standardize:
        scaler = StandardScaler()
        m2[['difference', 'Score']] = scaler.fit_transform(m2[['difference','Score']])
    corr = np.corrcoef(m2.difference, m2.Score)
    pr = pearsonr(m2.difference, m2.Score)
    spearmancorr = m2[['difference','Score']].corr(method='spearman')
    if include_corrs:
        title = (
            f'{renamed_model_name}, '
            f'Spearman: {spearmancorr.iloc[0,1]:.3f}\n'
            f'Pearson: {corr[0, 1]:.3f}, p-val: {pr[1]:.3f}'
        )
    else:
        title = f'Model: {renamed_model_name}'
    plot_stdized_wlabels(m2, axes[idx], title, alpha=0.7)

for i in range(num_models, num_rows * 2):
    fig.delaxes(axes[i])

plt.tight_layout()
if include_corrs:
    plt.savefig(plotdir / "vals_scatter_all_labels_wcorrs_minoverplot.pdf")
else:
    plt.savefig(plotdir / "vals_scatter_all_labels_minoverplot.pdf")
