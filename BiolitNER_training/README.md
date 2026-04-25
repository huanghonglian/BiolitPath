**Download RoBERTa-large-PM-M3-Voc-hf**


```python
wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz
tar -zxvf RoBERTa-large-PM-M3-Voc-hf.tar.gz
```

**Download Dataset**

automatically annotate articles from PubMed  
[NERdata_pubmed](https://drive.google.com/file/d/1T6BzHivhyLs-kb87LftjFOAN6oMoZY_E/view?usp=drive_link)  
gold-standard dataset  
[NERdata](https://drive.google.com/file/d/1AcQe8LpyPG1g9Ibpipr0WCNxJSEInAM1/view?usp=drive_link)

**Training**  
This approach uses a two-stage fine-tuning strategy based on a pre-trained RoBERTa model.

(1)Stage 1 Fine-tuning  

In the first stage, the pre-trained RoBERTa model is preliminarily fine-tuned on the automatically annotated pubmed data (NERdata_pubmed).


```python
bash train_first.sh
```

(2)Stage 2 Fine-tuning

In the second stage, the model is further fine-tuned on the gold-standard dataset (NERdata). 


```python
bash train_second.sh
```

**Evaluation**

We evaluate the model after the second-stage fine-tuning, referred to as **BiolitNER**.  
The following example uses the NCBI-disease test set for evaluation:


```python
bash eval.sh
```

To evaluate on other datasets, modify the `ENTITY` and `EVAL_TYPE` variables accordingly.

**Reproduce our evaluation results**

To evaluate the model's predictions at a consistent granularity, we assess performance at the word level instead of the token-level evaluation described above.


```python
cd evaluation
python bio_eval.py
```
