import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from modelscope import ClapModel, ClapProcessor

librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_sample = librispeech_dummy[0]

model = ClapModel.from_pretrained("/data/250010161/jiabao/models/CLAP").to(0)
processor = ClapProcessor.from_pretrained("/data/250010161/jiabao/models/CLAP")

inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="pt", sample_rate=16000).to(0)
audio_embed = model.get_audio_features(**inputs)

print(audio_embed.shape) # torch.Size([1, 512])
print(audio_embed)