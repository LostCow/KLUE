# KLUE MRC

## Train

```bash
sh finetune_retro_reader_mrc.sh
```

## Inference

```
# intensive reader model
tar -czvf intensive.tar.gz config.json pytorch_model.bin special_tokens_map.json tokenizer_config.json vocab.txt tokenizer.json

# sketch reader model
tar -czvf sketch.tar.gz config.json pytorch_model.bin special_tokens_map.json tokenizer_config.json vocab.txt tokenizer.json

# submission files
tar -czvf klue-mrc.tar.gz model arguments.py inference.py model.py processor.py retro_reader.py utils_qa.py requirements.txt
```

