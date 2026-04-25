**Download SapBERT-from-PubMedBERT-fulltext**

Download the model files from the following link:  
[https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)  
Save all downloaded files into a local directory named: `SapBERT-from-PubMedBERT-fulltext` 

**Download Dataset**

[datasets](https://drive.google.com/file/d/1QRt8wST9AHjRZnXZLuC6gpOOftr0-D8O/view?usp=drive_link)

**Training**  


```python
bash train.sh
```

**Evaluation**


```python
bash eval.sh
```

To evaluate on other datasets, modify the `DATA_DIR` variables accordingly.
