# News source analysis replication

To generate the results for news sources:

0. Get the News source human coded evaluation data from [https://github.com/IgniparousTempest/mediabiasfactcheck.com-bias/tree/master](https://github.com/IgniparousTempest/mediabiasfactcheck.com-bias/tree/master)

1. Generate the results for each model using `1_gen_news.py`, e.g. python -m src.news.1_gen_news --model gemini-1.5-flash

2. Code these outputs according to the 3-value coding schema using `src/news/2_code_news.py`
    - This involves submitting batches to openai, waiting (max 24 hours, generally only 1-2h), then processing the outcome.
    - e.g. Run this interactively using ipython to be able to monitor when batchs are done.

3. Run regressions and generate tables and plots using `3_news_logit.py` (includes summary stats) and `3a_news_ols.py`

