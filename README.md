# [Goodreads Datasets](https://mengtingwan.github.io/data/goodreads.html)

#### NOTE: Our datasets have been moved! Please see our new [webpage](https://mengtingwan.github.io/data/goodreads.html) about how to download these datasets.

The datasets were collected in late 2017 from [goodreads](https://goodreads.com). Details of the datasets are described in the [dataset website](https://mengtingwan.github.io/data/goodreads.html)

**We collected these datasets for academic use only! Please do not redistribute them or use for commercial purposes.**

## Citations
If you are using our dataset, please cite the following papers:

- Mengting Wan, Julian McAuley, "[Item Recommendation on Monotonic Behavior Chains](https://github.com/MengtingWan/mengtingwan.github.io/raw/master/paper/recsys18_mwan.pdf)", in RecSys'18. [[bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/recsys/WanM18)]
- Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "[Fine-Grained Spoiler Detection from Large-Scale Review Corpora](https://github.com/MengtingWan/mengtingwan.github.io/raw/master/paper/acl19_mwan.pdf)", in ACL'19. [[bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/acl/WanMNM19)]



## Project: Book Preference Prediction

This project uses the Goodreads dataset to study two related but distinct problems:

### Rating Prediction (Baselines)

`scripts/train.py` and `scripts/hyperparameter_tuning.py` train collaborative filtering and content-based models to **predict a user's numerical rating** for a book. These are evaluated on held-out ratings from the interaction matrix.

### Preference Prediction (LLM Evaluation)

`util/run_llm_eval.py` evaluates an LLM on a **pairwise preference prediction** task: given a user's past review(s) and community reviews for two candidate books, predict which book the user will prefer (i.e., rate higher). This is a different framing — rather than predicting an absolute rating, the model ranks two options.

The dataset for this task lives in `util/book_preference_dataset.jsonl` (205 entries). Each entry contains:
- A reference review written by the user (with rating)
- Community reviews for two candidate books (Book A and Book B)
- Ground truth: which book the user rated higher, and by how much (`rating_difference`)

**Metrics:**
- `accuracy`: fraction of entries where the LLM picked the correct book
- `preference_score`: mean signed rating difference (positive = better than random, 0 = random, negative = worse)

**Usage:**
```bash
python util/run_llm_eval.py \
    --input util/book_preference_dataset.jsonl \
    --output_dir <output_dir> \
    --model Qwen/Qwen2.5-7B-Instruct   # or any HF causal LM
```

Or submit via SLURM: `sbatch scripts/run_llm_eval.sh`

## Notebooks/Code Samples

We've created several notebooks (in python 3.7) to illustrate how to download/read these datasets, and provide some basic explorations of the data.

- [download.ipynb](/download.ipynb): If you prefer to download datasets without GUI. This notebook will show how to download files in bash/python.
- [samples.ipynb](/samples.ipynb): This notebook will show how to read '.json.gz' files line-by-line and display sample records of each file.
- [statistics.ipynb](/statistics.ipynb): This notebook will calculate some basic statistics of the datasets (except the largest complete interaction file 'goodreads_interactions.csv'). Running this notebook may take a while.
- [distributions.ipynb](/distributions.ipynb): This notebook will operate on the complete interaction file 'goodreads_interactions.csv' and provide some explorations of the distributions of these interactions. **Note: Run this notebook only when you have LARGE memory (recommend 32g+)!!**
- [reviews.ipynb](/reviews.ipynb): This notebook will calculate some statistics of the review datasets.

