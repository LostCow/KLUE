# KLUE STS

## Train

```bash
python train.py
```

## Inference

```
# STS Model
tar -czvf klue-sts.tar.gz config.json pytorch_model.bin special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt

# submission files
tar -czvf klue-sts.tar.gz dataset.py inference.py model.py model requirements.txt utils.py 
```