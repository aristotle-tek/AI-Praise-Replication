



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
import seaborn as sns
from sklearn.preprocessing import StandardScaler


from pathlib import Path


base_path = Path(__file__).resolve().parent.parent


plotdir = base_path / "data" / "values" / "plots"
coded_dir = base_path / "data" / "values" / "coded"



model_list = ['gpt-3.5-turbo', 'claude-3-sonnet-20240229', 'gemini-1.5-flash', \
'qwen1.5-32b-chat','meta-llama-3-70b-instruct',"mixtral-8x22b-instruct"]




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




#------------------------
# Plots by 12 categories
#------------------------


all_results = []

for whichmodel in model_list:
    print(whichmodel)
    jdf = pd.read_csv(coded_dir / f"Coded_outputs_{whichmodel}_sents_wcodes.csv")

    category_results = {'Category': [], 'Pearson_r': [], 'Pearson_p': [], 'Pearson_lower': [], 'Pearson_upper': [],
                        'Spearman_rho': [], 'Spearman_p': [], 'Spearman_lower': [], 'Spearman_upper': [], 'Engagement': []}

    categories = jdf['category12'].unique()

    for cat in categories:
        cat_data = jdf[jdf['category12'] == cat]

        x = cat_data['difference']
        y = cat_data['Score']

        #  Engagement = proportion of non-zero responses 
        non_zero_responses = (cat_data['praisecode'] != 0).sum()
        total_responses = len(cat_data)
        engagement_score = non_zero_responses / total_responses if total_responses > 0 else np.nan

        mask = pd.notnull(x) & pd.notnull(y)
        x = x[mask]
        y = y[mask]

        n = len(x)
        if n >= 4:
            r, p_value = pearsonr(x, y)

            # Fisher z-transformation for CIs
            z = np.arctanh(r)
            SE = 1 / np.sqrt(n - 3)
            z_critical = norm.ppf(0.975)  # 95% CI

            z_lower = z - z_critical * SE
            z_upper = z + z_critical * SE

            # Transform back to r
            r_lower = np.tanh(z_lower)
            r_upper = np.tanh(z_upper)

            rho, sp_p_value = spearmanr(x, y)

            # Bootstrap for Spearman CIs
            n_bootstraps = 1000
            bootstrapped_rhos = []
            for i in range(n_bootstraps):
                indices = np.random.randint(0, n, n)
                x_sample = x.iloc[indices]
                y_sample = y.iloc[indices]
                rho_sample, _ = spearmanr(x_sample, y_sample)
                bootstrapped_rhos.append(rho_sample)

            # CIs from percentiles
            lower_percentile = 2.5
            upper_percentile = 97.5
            rho_lower = np.percentile(bootstrapped_rhos, lower_percentile)
            rho_upper = np.percentile(bootstrapped_rhos, upper_percentile)
        else:
            r = np.nan
            p_value = np.nan
            r_lower = np.nan
            r_upper = np.nan

            rho = np.nan
            sp_p_value = np.nan
            rho_lower = np.nan
            rho_upper = np.nan

        category_results['Category'].append(cat)
        category_results['Pearson_r'].append(r)
        category_results['Pearson_p'].append(p_value)
        category_results['Pearson_lower'].append(r_lower)
        category_results['Pearson_upper'].append(r_upper)
        category_results['Spearman_rho'].append(rho)
        category_results['Spearman_p'].append(sp_p_value)
        category_results['Spearman_lower'].append(rho_lower)
        category_results['Spearman_upper'].append(rho_upper)
        category_results['Engagement'].append(engagement_score)
    cat_results_df = pd.DataFrame(category_results)
    cat_results_df['Model'] = whichmodel

    all_results.append(cat_results_df)

all_results_df = pd.concat(all_results, ignore_index=True)

all_results_df = all_results_df.dropna(subset=['Pearson_r', 'Pearson_lower', 'Pearson_upper'])



#-------------------------------------
# aside - is there a relationship between engagement and correlation?
#-------------------------------------

import statsmodels.formula.api as smf

all_results_df['Model'] = all_results_df['Model'].astype('category')
model = smf.ols(formula="Spearman_rho ~ Engagement + C(Model)", data=all_results_df).fit()
print(model.summary())


"""
                                            coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------------------
Intercept                                 0.1111      0.152      0.729      0.468      -0.193       0.415
C(Model)[T.gemini-1.5-flash]              0.0233      0.086      0.271      0.787      -0.148       0.195
C(Model)[T.gpt-3.5-turbo]                 0.1243      0.085      1.460      0.149      -0.046       0.294
C(Model)[T.meta-llama-3-70b-instruct]    -0.0417      0.086     -0.486      0.629      -0.213       0.130
C(Model)[T.mixtral-8x22b-instruct]       -0.0043      0.085     -0.051      0.960      -0.174       0.165
C(Model)[T.qwen1.5-32b-chat]             -0.0069      0.085     -0.081      0.935      -0.177       0.163
Engagement                                0.2019      0.196      1.033      0.306      -0.189       0.593
"""


#-------------------------------------
# plot separately, with cat names
#-------------------------------------
# dict to shorten names
category_mapping = {
    'Apologies and Blame': 'Blame',
    'Communication': 'Communication',
    'Consumption and Habits': 'Consumption',
    'Helping and Saving': 'Helping',
    'Lies and Misinformation': 'Misinformation',
    'Physical Affection and Relationships': 'Relationships',
    'Politeness and Social Norms': 'Norms',
    'Self-Improvement and Morality': 'Self-Improvement',
    'Theft and Dishonesty': 'Dishonesty',
    'Violence and Harm': 'Violence',
    'Waste and Misuse': 'Waste',
    'Weapons and Safety': 'Weapons'
}


colors = plt.cm.tab10(range(len(model_list)))
model_color_map = dict(zip(model_list, colors))


# Plot Corr vs. Engagement
fig, ax = plt.subplots(figsize=(12, 7))


# LaTeX font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Source Serif Pro"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

for model in model_list:
    model_data = all_results_df[all_results_df['Model'] == model]
    x = model_data['Spearman_rho']
    y = model_data['Engagement']
    categories = model_data['Category'].map(category_mapping)
    
    for xi, yi, category in zip(x, y, categories):
        ax.text(xi, yi, category, ha='center', va='center', color=model_color_map[model])

#ax.set_title('')
ax.set_xlabel('Spearman Correlation')
ax.set_ylabel('Engagement')
ax.set_xlim(-0.4, 0.9)
ax.set_ylim(0.25, 1.0)
ax.legend(handles=[plt.Line2D([0], [0], color=model_color_map[model], marker='o', linestyle='', markersize=10, label=model) for model in model_list], loc='lower right')
plt.tight_layout()
plt.savefig(plotdir / 'spearman_correlation_vs_engagement.pdf')
plt.close()




# version 2 - remove overplots...

name_map = {
    'qwen1.5-32b-chat': "Qwen-1.5-32b",
    'gpt-3.5-turbo': "GPT-3.5-turbo",
    'meta-llama-3-70b-instruct': "Llama-3 70B",
    'gemini-1.5-flash': "Gemini-1.5-flash",
    "mixtral-8x22b-instruct": "Mixtral-8x22b",
    'claude-3-sonnet-20240229': "Claude-3-sonnet"
}


from matplotlib.transforms import Bbox

def overlap_fraction(bbox1, bbox2):
    # calculate a fraction of overlap between two bounding boxes
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


fig, ax = plt.subplots(figsize=(12, 7))

used_bboxes = []
overlap_tolerance = 0.3  # allow up to 30% overlap

for model in model_list:
    model_data = all_results_df[all_results_df['Model'] == model]
    x = model_data['Spearman_rho']
    y = model_data['Engagement']
    categories = model_data['Category'].map(category_mapping)

    for xi, yi, category in zip(x, y, categories):
        label = ax.text(xi, yi, category, ha='center', va='center', fontsize=14, color=model_color_map[model])
        bbox = label.get_window_extent(renderer=fig.canvas.get_renderer())

        # check overlap fraction
        too_much_overlap = False
        for existing_bbox in used_bboxes:
            if overlap_fraction(bbox, existing_bbox) > overlap_tolerance:
                too_much_overlap = True
                break

        if not too_much_overlap:
            used_bboxes.append(bbox)
        else:
            label.remove()

ax.set_xlabel('Spearman Correlation')
ax.set_ylabel('Engagement')
ax.set_xlim(-0.4, 0.9)
ax.set_ylim(0.25, 1.0)

ax.legend(
    handles=[
        plt.Line2D([0], [0], color=model_color_map[m], marker='o', linestyle='', markersize=10, label=name_map.get(m, m))
        for m in model_list
    ], 
    loc='lower right'
)

plt.tight_layout()
plt.savefig(plotdir / 'spearman_correlation_vs_engagement_min_overplot.pdf')
plt.close()
