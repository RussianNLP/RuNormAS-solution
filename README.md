# Solution of RuNormAS
Solution of [RuNormAS competition](https://github.com/dialogue-evaluation/RuNormAS) in [Dialogue2021](http://www.dialog-21.ru/evaluation/).

Solution based on RuGPT3XL. Model was tuned on train data. The best model was trained on train data and lenta news.

Solution is archived **0.96452** (generic)	**0.95750** (named) accuracy as mesured by organizers. But if exclude evaluation errors our best model is archived **0.976700** (generic)	**0.980977** (named) accuracy
## Usage
Example of usage you can see [here](Usage.ipynb)

Pretrained model [here](https://disk.yandex.ru/d/g5bVLObqtKCxcA)
## Prepare solution
Before run all code make the following dirs:
```bash
mkdir ../models/xl
mkdir ../data
mkdir ../test_pred
```
### Download data
Download data for train.
```bash
/bin/bash ./scripts/get_data_from_repo.sh ../data/
```
Note! Here no code for download lenta news.

### Process data for LM
Read data and make files for LM (for best solution).
```bash
/bin/bash ./scripts/process_data_from_repo.sh
```

Split data on train and valid for validation.

```bash
python modules/data/split_data.py
```

### Run training
#### Install env

```bash
sh scripts/install_env_for_gpt.sh
```

#### Run finetuning
Finetune RuGPT3XL.

```bash
cd scripts
sh deepspeed_xl_runormas_v14_130k_finetune.sh
```

### Predict for competition
#### Prepare baseline
First of all make prediction for baseline.

```bash
python baseline.py
```

#### Predict with model
Run the following commands:

```bash
cd scripts
sh deepspeed_xl_runormas_v14_130k_finetune.sh
sh xl_runormas_pred_distributed_v14_130k_finetune_10f.sh
```

Script `xl_runormas_pred_distributed_v14_130k_finetune_10f.sh` is needed for predict on 10 files that was not predicted with `deepspeed_xl_runormas_v14_130k_finetune.sh` (this files was trancated while distributed generation).

#### Post-processing for submission
For post-processing run the following notebook [make_prediction.ipynb](make_prediction.ipynb). This step improve accuracy around 1-2%.

Make archive for submission:

```bash
cd ../test_pred/v14_130k_finetune_fixed
7z a submission.zip *
```

## Error analysis
Also we add error_analysis [notebook](error_analysis.ipynb) of our best model.
