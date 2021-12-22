# KLUE RE

## Train

## Inference

```
# RE Model
tar -czvf klue-re.tar.gz added_tokens.json config.json pytorch_model.bin special_tokens_map.json tokenizer_config.json vocab.txt 

# submission files
tar -czvf klue-re.tar.gz model dataset.py inference.py losses.py model.py utils.py requirements.txt 
```