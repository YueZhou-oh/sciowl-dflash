# SciOWL DFlash 

compare speed and accuracy between direct inference and speculative decoding

## contents

- `scripts/serve_baseline.sh`: run SciOWL
- `scripts/serve_dflash.sh`: run SciOWL with DFlash draft。
- `scripts/test_parallel.py`: run parallel testing
- `scripts/test.jsonl`: six different molecular property prediction tasks
- `ckpt/dflash_epoch4/`: draft model for speculative decoding

## script

start server and run scientific testing

```bash
bash sciowl_dflash_infer_pack/scripts/serve_baseline.sh
# or
bash sciowl_dflash_infer_pack/scripts/serve_dflash.sh

bash test_parallel.py --output logs/speed_baseline_predictions.jsonl

python score_mol_instructions_new.py --data-path logs/test.jsonl --predictions logs/speed_baseline_predictions.jsonl 
```

