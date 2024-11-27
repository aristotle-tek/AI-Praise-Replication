# Readme - Values analysis

Schramowski, et al. data from here [https://github.com/ml-research/MoRT_NMI/tree/master](https://github.com/ml-research/MoRT_NMI/tree/master)
It is also archived on [Zenodo](https://doi.org/10.5281/zenodo.5906596)


## 0. Modifications to Schramowski, et al. data

First, we created a modified version of the data, in which I made minor revisions to the action descriptions (for example, changing "help coworkers" to "help my coworkers") and created morally opposite versions of the action. For example, for the action "to be a bad person" the opposite action is "to be a good person".

See `/data/values/prompts_userglobal.xlsx`


---
## 1. Create prompts

Run `1_create_prompts.py`. This file adds the creates prompts, saves as `batch_data_01_df.csv`
- This also submit batch for OpenAI's gpt-3.5 (in lieu of 2)


---
## 2. Call LLM APIs

`/src/values/2_call_LLMs.py` 

---
## 3. Label LLM outputs

(for more details see example in `src/values/3_code_values.py` )

---
## 4. analyze results


`/src/values/4_vals_scatterplots.py` generates:
- `vals_scatter_all_labels.pdf`


`/src/values/4_plot_corr-v-engagement.py` generates:
- `spearman_correlation_vs_engagement.pdf`




--------------------
### Data
--------------------



1. Schramowski data with additional column: data/vals/prompts_userglobal.xlsx"

2. Categorized - based on GPT...
categories_file = `data/vals/vals_categorized.csv`

3. LLM responses - `data/vals/batchfiles/`

Batchfiles submitted for LLM evaluation: includes the original prompt, the LLM response, and the evaluation prompt.

format: `"values_" + whichmodel + "_"+ whichdata + "01.csv"`
(whichdata in 'sents','sentsrev')

4. Coded, assembled data for plots

Folder: `data/vals/coded/`

format: `coded_dir + f"Coded_outputs_{whichmodel}_sents_wcodes.csv")`





