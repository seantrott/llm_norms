# LLM Norms

Probing whether LLMs can augment psycholinguistic datasets.

## The data

All data can be found in `data`.

- `data/raw`: contains original human norms for each judgment task, as well as the instructions.
- `data/processed`: contains output of GPT-4 norming process.  
- `data/lexical_statistics`: contains files needed to reproduce the *substitution analyses*. 

The LLM-generated norms are already included, but if you'd like to regenerate them, see the section below.

## Reproducing the norms

The norms can be reproduced using either `src/models/similarity.py` (for judgments comparing two words or two contexts) or `src/models/single_word.py` (for judgments involving a single word). The task itself can be modified in the `__main___` part of the script, or using a command line argument.

Note that access the OpenAI API will require an [**authentication**](https://openai.com/blog/openai-api). The code assumes this authentication information from a file in `src/models` called `gpt_key`; to run the code, you'll need to create an analogous file with your own authentication information.

### Processing the output

Once you've reproduced the norms, you can run `src/processing/process_datasets.py`, which will convert `.txt` files to `.csv` files.

## Running the analyses

Finally, the relevant analyses are contained in `src/analysis` as `.Rmd` files. The results have already been **knit** to `.html`.

