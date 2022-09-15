# How to test

- Set the appropriate paths in `prep/test_prep.sh` and run it to get the preprocessed test data `test_id.txt`.
- Set the appropriate paths in `src/test.py` and run it to get the perplexity.

There should be no unknown tokens, and for the number of tokens one could simply use the `wc` command on `test_id.txt`.