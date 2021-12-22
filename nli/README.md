# KLUE MRC

## Train

```bash
python train.py
```

## Inference

```
# NLI Model
tar -czvf klue-nli.tar.gz config.json pytorch_model.bin special_tokens_map.json tokenizer_config.json vocab.txt tokenizer.json

# submission files
tar -czvf klue-nli.tar.gz model inference.py dataloader.py dataset.py requirements.txt utils.py
```

