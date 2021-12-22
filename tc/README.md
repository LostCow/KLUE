# KLUE TC

## Train

```bash
python train.py
```

## Inference

```
# TC Model
tar -czvf ynat.tar.gz config.json pytorch_model.bin special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt

# submission files
tar -czvf sub.tar.gz dataset.py model.py requirements.txt utils.py dataloader.py inference.py loss.py model
```