# Readme - International Politicians analysis


## Data


- The list of politicians was drawn from this [Wikipedia page on current heads of government](https://en.wikipedia.org/wiki/List_of_current_heads_of_state_and_government), converted to csv, and supplemented with a few other leaders. For the full list, see the intl_leaders_file: `data/world_leaders/intl_pol_leaders_long.xlsx`


## Code


`1_gen_intl_pol.py` - call LLMs (aside from GPT-3.5, submitted as a batch as in the other experiments).

---
## Label LLM outputs

`2_code_intl_pol.py`



---
## Generate plot

- `pol_intl_plot.py`

Generates `intpol_selected_plot.pdf`

---
## regression

`3_regr_intl_plot.py`
