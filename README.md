# sentiment-classifier

Minimal text classification pipeline: TF-IDF (unigrams + bigrams) into logistic regression, with a train/predict CLI. Ships with a small synthetic sample review dataset (template-generated, for demo only) (`data/reviews.csv`) so it runs out of the box; swap in your own CSV with `text,label` columns.

```bash
pip install -r requirements.txt
python classifier.py train --data data/reviews.csv
python classifier.py predict "great quality, love it"
pytest
```
