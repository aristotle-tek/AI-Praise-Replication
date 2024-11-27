#3_pol_intl_plot.py




import re
import json
import numpy as np
import pandas as pd
import time
import getpass
import os
import asyncio
import logging
import uuid


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


import itertools
from scipy.stats import pearsonr, spearmanr, norm # older version -from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats

import seaborn as sns


from pathlib import Path


base_path = Path(__file__).resolve().parent.parent

code_path = base_path / "src"


eval_folder=base_path / "data" / "world_leaders" / "output" 
plot_folder=base_path / "data" / "world_leaders" / "plots" 



#models = ["claude", "gpt35",'gemini','mixtral','llama3','qwen1.5-32b-chat']
model_list = ['gpt-3.5-turbo', 'claude-3-sonnet-20240229', 'gemini-1.5-flash', \
'qwen1.5-32b-chat','meta-llama-3-70b-instruct',"mixtral-8x22b-instruct"]


# key to get from prompts to name
intl_leaders_file = base_path / "data" / "world_leaders" / "intl_pol_leaders_long.xlsx"

connect = pd.read_excel(intl_leaders_file)

leaders = list(connect.leader)
states = list(connect.state)
states = [state.replace('\xa0', '') for state in states]


leaders = [re.sub(r'^.*?[\u2013–\-]\s*', '', x.replace('\xa0', ' ')) for x  in leaders]


# LaTex font and transparency
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



# while we're at it - calc % non-zero
percentages = []

for whichmodel in model_list:
    df = pd.read_csv(eval_folder + "intpol_evalcoded_intlpol_eval_" + whichmodel + ".csv")
    df['leader'] = np.tile(leaders, int(np.ceil(len(df) / len(leaders))))[:len(df)]
    df['state'] = np.tile(states, int(np.ceil(len(df) / len(states))))[:len(df)]
    
    # Reverse scores on 2nd half
    df1 = df.iloc[:1990]
    df2 = df.iloc[1990:]
    df2['correctedcode'] = - df2['correctedcode']
    df = pd.concat([df1, df2])
    
    df = df[df.correctedcode != 999]
    df = df[df.correctedcode != -999]
    
    df1_nonzero = len(df1[df1.correctedcode != 0]) / len(df1) * 100
    df2_nonzero = len(df2[df2.correctedcode != 0]) / len(df2) * 100
    df_all_nonzero = len(df[df.correctedcode != 0]) / len(df) * 100
    
    percentages.append({
        'model': whichmodel,
        'df1_engagm_pct': df1_nonzero,
        'df2_engagm_pct': df2_nonzero,
        'df_all_engagm_pct': df_all_nonzero
    })


percentages_df = pd.DataFrame(percentages)
print(percentages_df)


latex_table = percentages_df.style.format({
    'df1_engagm_pct': '{:.1f}',
    'df2_engagm_pct': '{:.1f}',
    'df_all_engagm_pct': '{:.1f}'
}).to_latex(
    caption="Engagement Measure by task, Model",
    label="tab:nonzero_percentages",
    column_format="|l|c|c|c|",
    hrules=True
)

print(latex_table)


combined_df = pd.concat(results)

leader_scores = combined_df.groupby(['leader', 'model'])['correctedcode'].mean().reset_index()

overall_leader_scores = leader_scores.groupby('leader')['correctedcode'].mean().reset_index()


num_leaders = 8

top_5_leaders = overall_leader_scores.nlargest(num_leaders, 'correctedcode')['leader']
bottom_5_leaders = overall_leader_scores.nsmallest(num_leaders, 'correctedcode')['leader']



selected_leaders = pd.concat([bottom_5_leaders, top_5_leaders])


sorted_leaders = overall_leader_scores.set_index('leader').loc[selected_leaders].sort_values(by='correctedcode').index.tolist()



plot_data_filtered = leader_scores[leader_scores['leader'].isin(selected_leaders)]


# Ensure categorical order
plot_data_filtered['leader'] = pd.Categorical(plot_data_filtered['leader'], categories=sorted_leaders, ordered=True)


#--- Plot ---


# mod leader names:
display_sorted_leaders = [
    'Bashar al-Assad',
    'Marine Le Pen',
    'Steve Bannon',
    'Benjamin Netanyahu',
    'Boris Johnson',
    'Min Aung Hlaing',
    'Kim Jong Un',
    'Nigel Farage',
    'Naomi Klein',
    'ASEAN Secretary General \nKao Kim Hourn',
    'ICJ President \nNawaf Salam',
    'Naruhito',
    'Nikos Christodoulides',
    'Sheikh Hasina',
    'IPCC Chair \nJim Skea',
    'Choguel Kokalla Maïga'
]


plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})


offset_step = 0.1  # space between points for each model to prevent overplot

# Map each model to an index for calculating offsets
model_offsets = {model: (i-2) * offset_step for i, model in enumerate(model_list)}

# Apply the offsets to x-axis positions
plot_data_filtered['x_offset'] = plot_data_filtered.apply(
    lambda row: sorted_leaders.index(row['leader']) + model_offsets[row['model']],
    axis=1
)

plt.figure(figsize=(12, 8))

# colors = sns.color_palette("muted", n_colors=len(model_list))
# sns.set_palette(colors)
colors = sns.color_palette("colorblind", n_colors=len(model_list))
sns.set_palette(colors)

sns.scatterplot(data=plot_data_filtered, x='x_offset', y='correctedcode', hue='model', style='model', s=100)

# show leader names
plt.xticks(ticks=range(len(sorted_leaders)), labels=display_sorted_leaders, rotation=90)
plt.gcf().subplots_adjust(bottom=0.25)  # Add space for x-tick labels

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.xlabel('')
plt.ylabel('Mean Score')
plt.legend(title='Model',  loc='lower right') # bbox_to_anchor=(1.05, 1), # bbox - outside
plt.tight_layout()

output_pdf_file = plot_folder + "intpol_selected_plot.pdf" # (converted pdf to eps using gimp since transparency fails.)
plt.savefig(output_pdf_file, format='pdf', bbox_inches='tight')
plt.close()



#-----------------------------------------
# plot from selected states and the EU




combined_df.state.replace("US politician", 'US', inplace=True)
combined_df.state.replace('United States', "US", inplace=True)
combined_df.state.replace('United Kingdom', "UK", inplace=True)
combined_df.state.replace('French politician', "France", inplace=True)
print(combined_df['state'].value_counts().to_string())


selected_states = ['China', 'US', 'United Kingdom', \
    'European Union', 'UK', 'France',"United States"]


filtered_df = combined_df[combined_df['state'].isin(selected_states)]

filtered_df = filtered_df[filtered_df.leader != 'Jordan Bardella'] # not known, too late to scene.
filtered_df = filtered_df[filtered_df.leader != 'Boris Johnson'] # already in other plot.


filtered_df['state'].replace('US politician', "U.S.", inplace=True)
filtered_df['state'].replace('United States', "U.S.", inplace=True)
filtered_df['state'].replace('United Kingdom', "U.K.", inplace=True)
filtered_df['state'].replace('Facebook/ Meta', "(Meta)", inplace=True)
filtered_df['state'].replace('European Union', "E.U.", inplace=True)


filtered_df['leader'].replace('Charles III', 'King Charles III', inplace=True)


filtered_df['state'].value_counts()

# label with state and name
filtered_df['state_leader'] = filtered_df['state'] + ': ' + filtered_df['leader']

filtered_df = filtered_df.sort_values(by=['state', 'leader'])



offset_step = 0.1

unique_models = filtered_df['model'].unique()
model_offsets = {model: (i - len(unique_models) / 2) * offset_step for i, model in enumerate(unique_models)}

filtered_df['x_offset'] = filtered_df.apply(
    lambda row: sorted(filtered_df['state_leader'].unique()).index(row['state_leader']) + model_offsets[row['model']],
    axis=1
)

plt.figure(figsize=(15, 8))
colors = sns.color_palette("colorblind", n_colors=len(unique_models))
sns.set_palette(colors)

sns.scatterplot(data=filtered_df, x='x_offset', y='correctedcode', hue='model', style='model', s=100)

state_leaders = sorted(filtered_df['state_leader'].unique())
plt.xticks(ticks=range(len(state_leaders)), labels=state_leaders, rotation=75, fontsize=10)# plt.xticks(rotation=45, ha='right', fontsize=10)

plt.xlabel('')
plt.ylabel('Mean Score')
plt.legend(title='Model', loc='lower right') # outside - bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


output_pdf_file = plotdir + "intpol_selstates_plot.pdf"
plt.savefig(output_pdf_file, format='pdf', bbox_inches='tight')
plt.close()



