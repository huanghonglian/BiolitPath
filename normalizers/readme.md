# Neural normalizer

## How to index dictionary embeddings
```
# index 
CUDA_VISIBLE_DEVICES=0 python neural_normalizer.py \
    --model_name_or_path ../model/biolitNEN \
    --dictionary_path ../resources/normalization/dictionary/dict_geoname_202601.txt \
    --cache_dir ../resources/normalization/normalizers/neural_norm_caches

```
