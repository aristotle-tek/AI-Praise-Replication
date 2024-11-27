

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
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})




#------------------------------
# Scatterplot with line of best fit
#------------------------------ 


# Number of subplots and layout configuration for 2 columns
num_models = len(model_list)
num_rows = (num_models + 1) // 2  # Calculate rows needed for 2 columns
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
axes = axes.flatten()

standardize = True
include_corrs = True #False

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
    plot_stdized_wlabels(m2, axes[idx], title, alpha=0.7)  # Apply alpha for transparency

# Hide unused subplots if num_models is odd
for i in range(num_models, num_rows * 2):
    fig.delaxes(axes[i])

plt.tight_layout()
if include_corrs:
    plt.savefig(plotdir + "vals_scatter_all_labels_wcorrs.pdf")
else:
    plt.savefig(plotdir + "vals_scatter_all_labels.pdf")
#plt.savefig(plotdir + "vals_scatter_stdlabels.eps") # may fail transp...

