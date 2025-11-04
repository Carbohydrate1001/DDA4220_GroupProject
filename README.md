# DDA4220 Group Project
## Environment Setup
```
pip install -r requirements.txt
```
Checkpoints of CLAP can be accessed at [Official Huggingface Repo](https://huggingface.co/lukewys/laion_clap/tree/main)

After setup, run
```
python /data/250010161/jiabao/packages/CLAP/src/laion_clap/unit_test.py
```
This program will load tokenizer from RoBerta and load CLAP checkpoint to extract semantic embeddings from audio files and text.