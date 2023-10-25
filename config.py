"""
this is the configuration file for alpaca llama model training.
this model optimized for RTX 4090. for larger GPUs, increase some of these?
"""

class Config:
    # language selection
    LANGUAGES = ["bn", "en"]
    LANGUAGE = LANGUAGES[0]
    # this could actually be 5 but i like powers of 2
    MICRO_BATCH_SIZE = 2
    # batch size  
    BATCH_SIZE = 128
    # gradient accumation
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    # we don't always need 3 epoch
    EPOCHS = 3  
    # the Karpathy constant
    LEARNING_RATE = 3e-4  
    # 256 accounts for about 96% of the data
    CUTOFF_LEN = 256
    # lora configuration  
    LORA_R = 8
    LOAD_IN_8BIT = True
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    VAL_SET_SIZE = 2000
    WARMUP_STEPS = 100
    FP16 = True
    LOGGING_STEPS = 20
    EVAL_STEPS = 200
    SAVE_STEPS = 200
    SAVE_TOTAL_LILMIT = 3
    DEVICE_MAP = "auto"
    VAL_SET_SIZE = 2000
    TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]
    SAVE_STRATEGY = "steps"
    EVALUATION_STRATEGY = "steps"
    DATA_PATH = "./data/data_sample.json"
    OUTPUT_DIR = "lora-alpaca"
    # bangla mask language model
    PRETRAIN_MODEL_NAME = "OdiaGenAI/odiagenAI-bengali-base-model-v1"

    RUN_NAME = "alpaca_lora_bangla"

    INSTRUCTION = {
        "en_input" : "Below is an instruction that describes a task, paired with an input that provides \
                     further context. Write a response that appropriately completes the request.",
        "bn_input" : "নীচে একটি নির্দেশ রয়েছে যা একটি টাস্ক বর্ণনা করে, একটি ইনপুটের সাথে যুক্ত যা আরও প্রসঙ্গ সরবরাহ করে। \
                      একটি প্রতিক্রিয়া লিখুন যা যথাযথভাবে অনুরোধটি সম্পূর্ণ করে।",
        "en":"Below is an instruction that describes a task. Write a response that appropriately \
              completes the request.",
        "bn": "নীচে একটি নির্দেশ যা একটি টাস্ক বর্ণনা করে। একটি প্রতিক্রিয়া লিখুন যা যথাযথভাবে অনুরোধটি সম্পূর্ণ করে।",
    }