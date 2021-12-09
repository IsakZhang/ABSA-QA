# ABSA-QA

Source code for our paper "[Aspect-based Sentiment Analysis in Question Answering Forums](https://aclanthology.org/2021.findings-emnlp.390.pdf)" in EMNLP-Findings 2021.


## Requirements

- Python 3.7.4
- torch 1.6.0
- fastNLP 0.5.5
- nltk 3.5


## Data

The datasets are from: [Aspect Sentiment Classification Towards Question-Answering with Reinforced Bidirectional Attention Network](https://www.aclweb.org/anthology/P19-1345.pdf) (ACL 2019)

*Due to the unclear license of the original data, we do not include it here, pls download from the original link in the paper.


## Training and Evaluation

To train the model:

```python
python train_model.py --dataset $category
```

where `$category` in `[elec, beauty, bags]` denotes the dataset name, the default one is the `elec` dataset. 

The model with best performance on the dev dataset will be saved in `outputs/` and evaluated on the test data after the training phase is finished


## Citation

If the code is used in your research, please star our repo and cite our paper as follows:
```
@inproceedings{zhang-etal-2021-aspect,
    title = "Aspect-based Sentiment Analysis in Question Answering Forums",
    author = "Zhang, Wenxuan  and
      Deng, Yang  and
      Li, Xin  and
      Bing, Lidong  and
      Lam, Wai",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.390",
    pages = "4582--4591"
}
```
