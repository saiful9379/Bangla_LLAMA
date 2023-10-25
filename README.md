# Bangla_LLAMA


# Requirements
```
datasets==2.14.5
sentencepiece==0.1.99
transformers==4.34.0.dev0
bitsandbytes==0.41.1
loralib
peft==0.6.0.dev0

```
or 
```
pip install -q bitsandbytes
pip install -q datasets loralib sentencepiece
pip install -q git+https://github.com/huggingface/transformers.git
pip install -q git+https://github.com/huggingface/peft.git
```


# Data structure

Format the data structure into JSON Format, Following the below structure,

```
[
    {
        "instruction": "হাই! কেমন চলছে?",
        "input": "",
        "output": "আমি ভালো আছি. তোমার কি অবস্থা?"
    },
    .,
    .,
    .,
    {
        "instruction": "তুমি কোন স্কুলে যাও?",
        "input": "",
        "output": "আমি পিসিসিতে যাই।"
    }
]
```


# Configuraiton

check the configuration for ```config.py``` and set your configuration.

# Traning

Run the ```train.py``` file

```
python train.py
```

Interactive notebook file 
```
./examples/Bangla_llama_Lora_finetune_final.ipynb
```

# Evaluation

```
Not Yet Done
```

# Infenence

```
./examples/Bangla_llama_lora_inference.ipynb
```

# References

```
1. https://github.com/tatsu-lab/stanford_alpaca
2. https://colab.research.google.com/drive/1eWAmesrW99p7e1nah5bipn0zikMb8XYC
3. https://huggingface.co/OdiaGenAI
4. https://huggingface.co/docs/diffusers/training/lora
```





