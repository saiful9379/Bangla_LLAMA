"""
Alpaca LLAMA Fine Tunning Model training script
"""


import os
import sys
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\n Please reinstall it: pip uninstall transformers && \pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from config import Config as cfg

device_map = cfg.DEVICE_MAP
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
GRADIENT_ACCUMULATION_STEPS  = cfg.GRADIENT_ACCUMULATION_STEPS
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size


class BanglaAlpacaLocraLLAMA:
    def __init__(self):
        self.lora_r = cfg.LORA_R
        self.lora_dropout = cfg.LORA_DROPOUT
        self.target_modules = cfg.TARGET_MODULES
        self.load_in_8bit= cfg.LOAD_IN_8BIT
        self.lora_alpha = cfg.LORA_ALPHA
        self.pretrain_model_name = cfg.PRETRAIN_MODEL_NAME
        self.cutoff_len = cfg.CUTOFF_LEN
        self.instruction = cfg.INSTRUCTION
        self.data_path = cfg.DATA_PATH
        self.language =  cfg.LANGUAGE
        self.warmup_steps = cfg.WARMUP_STEPS
        self.num_train_epochs = cfg.EPOCHS
        self.learning_rate = cfg.LEARNING_RATE
        self.fp16 = cfg.FP16
        self.save_strategy= cfg.SAVE_STRATEGY
        self.eval_steps=cfg.EVAL_STEPS 
        self.save_steps =cfg.SAVE_STEPS
        self.output_dir=cfg.OUTPUT_DIR
        self.save_total_limit=cfg.SAVE_TOTAL_LILMIT
        self.per_device_train_batch_size = cfg.MICRO_BATCH_SIZE
        self.logging_steps=cfg.LOGGING_STEPS
        self.evaluation_strategy = cfg.EVALUATION_STRATEGY
        self.run_name = cfg.RUN_NAME
        self.val_set_size = cfg.VAL_SET_SIZE
        self.tokenizer = self.llm_tokenizer_loading()

    def llm_model_loading(self)-> object:
        """
        pretrain model loading and peft configuration
        """

        model = LlamaForCausalLM.from_pretrained(
            self.pretrain_model_name,
            load_in_8bit=self.load_in_8bit,
            device_map=device_map
        )
        model = prepare_model_for_int8_training(model)

        # print("lora_r", type(self.lora_r))

        config = LoraConfig(
            r = self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
      
        return model

    def llm_tokenizer_loading(self)->object:
        """
        pretrain tokenizer loading
        """
        tokenizer = LlamaTokenizer.from_pretrained(
            self.pretrain_model_name, 
            add_eos_token=True
        )
        tokenizer.pad_token_id = 0
        return tokenizer
        
    def tokenize(self, prompt:str)->dict:
        """
        tokenizer way of prompt
        there's  a way to do this with the tokenizer settings but again, gotta move fast
        """
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len + 1,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }

    def generate_prompt(self, data_point:dict, language:dict)-> str:
        """
        prompt generation
        """

        if data_point["input"]:
            format_data = f"""{self.instruction[f"{language}_input"]}
            ### Instruction:
            {data_point["instruction"]}
            ### Input:
            {data_point["input"]}
            ### Response:
            {data_point["output"]}
            """
        else:
            format_data = f"""{self.instruction[language]}
            ### Instruction:
            {data_point["instruction"]}
            ### Response:
            {data_point["output"]}
            """
        return format_data

    def generate_and_tokenize_prompt(self, data_point:dict)-> list:
        """
        prompt and tokenizer generation process
        """
        prompt = self.generate_prompt(data_point, self.language)
        return self.tokenize(prompt)

    def data_processing(self):
        """
        data processing file
        input data structure format[JSON] :
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
        """
        data = load_dataset("json", data_files=self.data_path)
        if self.val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42
            )
            train_data = train_val["train"].shuffle().map(self.generate_and_tokenize_prompt)
            val_data = train_val["test"].shuffle().map(self.generate_and_tokenize_prompt)
        else:
            train_data = data["train"].shuffle().map(self.generate_and_tokenize_prompt)
            val_data = None
        return train_data, val_data

    def training(self):
        """
        Training process function
        """
        print("==============  Start Data Processing =============")
        train_data, val_data = self.data_processing()

        print("Training Data   : ", len(train_data))
        print("Validation Data : ", len(val_data))
        print("=================== Complete=======================")
        model = self.llm_model_loading()
        # define the training arguments
        trainer = transformers.Trainer(
            model= model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.per_device_train_batch_size,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                warmup_steps=self.warmup_steps,
                num_train_epochs=self.num_train_epochs,
                learning_rate=self.learning_rate,
                fp16=self.fp16,
                logging_steps=self.logging_steps,
                evaluation_strategy= self.evaluation_strategy if self.val_set_size > 0 else "no",
                save_strategy= self.save_strategy,
                eval_steps=self.eval_steps if self.val_set_size > 0 else None,
                save_steps=self.save_steps,
                output_dir=self.output_dir,
                save_total_limit=self.save_total_limit,
                load_best_model_at_end=True if self.val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                # report_to="wandb",  # enable logging to W&B
                run_name= self.run_name
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        model.config.use_cache = False

        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

        if torch.__version__ >= "2" and sys.platform != "win32":
          model = torch.compile(model)


        trainer.train()
        model.save_pretrained(self.output_dir)


if __name__ == "__main__":

    bnal_llama = BanglaAlpacaLocraLLAMA()
    bnal_llama.training()