# TransDrift: Modeling Word-Embedding Drift using Transformer

This repository is the official implementation of [TransDrift](https://arxiv.org/abs/2206.08081).

##  Training TransDrift model

To train the TransDrift model mentioned in the paper update script/arch.py, run this command:

```
python3 script/transDrift.py
```

## Performing downstream task

To perform downstream task mentioned in the paper for amazon dataset update downstream/amazon/arch.py, run this command:

```
python3 downstream/amazon/downstream.py
```

To perform downstream task mentioned in the paper for yelp dataset update downstream/yelp/arch.py, run this command:

```
python3 downstream/yelp/downstream.py
```


