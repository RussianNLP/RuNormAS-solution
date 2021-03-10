### Download data
```bash
/bin/bash ./scripts/get_data_from_repo.sh ../data/
```

### Process data for LM
Read data and make files for LM.
```bash
/bin/bash ./scripts/process_data_from_repo.sh
```

Split data on train and valid.

```bash
python modules/data/split_data.py
```

### Run training
Instructions for christophari.
#### Install env

```bash
sh scripts/install_env_for_gpt.sh
```

#### Run finetuning
Finetune RuGPT3XL.

```bash
cd scripts
sh deepspeed_xl_runormas.sh
```
