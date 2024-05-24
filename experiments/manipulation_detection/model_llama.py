import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback
)
from threading import Thread
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os
from datasets import Dataset


class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logging.info(f"Epoch: {int(state.epoch)}")
            logging.info(f"Best checkpoint: {state.best_model_checkpoint}, Best f1: {state.best_metric}")
            logging.info(logs)


class LlamaModel:
    def __init__(self,
                 model,
                 load_from_local,
                 temperature,
                 top_p,
                 top_k,
                 repetition_penalty,
                 max_new_tokens,
                 max_input_token_length,
                 ft_output_dir):
        if load_from_local:
            self.model_id = model
        else:
            self.model_id = "meta-llama/" + model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id,
                                                          torch_dtype=torch.float16,
                                                          device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, verbose=False)
        self.tokenizer.use_default_system_prompt = False
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.max_input_token_length = max_input_token_length
        self.ft_output_dir = ft_output_dir
        if not os.path.exists(self.ft_output_dir):
            os.makedirs(self.ft_output_dir)

    def zeroshot_prompting(self, dialogue):
        conversation = []
        system_prompt = """I will provide you with a dialogue. Please determine if it contains elements of mental manipulation. Just answer with 'Yes' or 'No', and don't add anything else.\n"""
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": dialogue})

        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt")
        if input_ids.shape[1] > self.max_input_token_length:
            input_ids = input_ids[:, -self.max_input_token_length:]
        input_ids = input_ids.to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            {"input_ids": input_ids},
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,  # generation length
            do_sample=False,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            num_beams=1,
            repetition_penalty=self.repetition_penalty,
        )

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
        res = ''.join(outputs)

        logging.info(system_prompt)
        logging.info(dialogue)
        logging.info('')
        logging.info(res)
        logging.info('')

        if res.lower().startswith('yes'):
            return 1
        elif res.lower().startswith('no'):
            return 0
        else:
            if 'yes' in res.lower():
                return 1
            elif 'no' in res.lower():
                return 0
            else:
                logging.info('Error: response of Llama is neither yes nor no.')
                return -1

    def fewshot_prompting(self, manip_examples, nonmanip_examples, dialogue):
        conversation = []
        total_example_num = len(manip_examples) + len(nonmanip_examples)
        system_prompt = f"""I will provide you with a dialogue. Please determine if it contains elements of mental manipulation. Just answer with 'Yes' or 'No', and don't add anything else. Here are {total_example_num} examples:\n"""
        conversation.append({"role": "system", "content": system_prompt})

        count_example = 0
        example_list = []
        for idx, row in manip_examples.iterrows():
            count_example += 1
            example = [
                {"role": "user", "content": f"Example {count_example}:\n{row['Dialogue']}"},
                {"role": "assistant", "content": "Yes"},
            ]
            example_list.extend(example)
        for idx, row in nonmanip_examples.iterrows():
            count_example += 1
            example = [
                {"role": "user", "content": f"Example {count_example}:\n{row['Dialogue']}"},
                {"role": "assistant", "content": "No"},
            ]
            example_list.extend(example)

        conversation += example_list
        conversation.append({"role": "user", "content": dialogue})

        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt")
        if input_ids.shape[1] > self.max_input_token_length:
            input_ids = input_ids[:, -self.max_input_token_length:]
        input_ids = input_ids.to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            {"input_ids": input_ids},
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,  # generation length
            do_sample=False,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            num_beams=1,
            repetition_penalty=self.repetition_penalty,
        )

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
        res = ''.join(outputs)

        logging.info(system_prompt)
        for example in example_list:
            logging.info(example['content'])
        logging.info('')
        logging.info(dialogue)
        logging.info('')
        logging.info(res)
        logging.info('')

        if 'yes' in res.lower():
            return 1
        elif 'no' in res.lower():
            return 0
        else:
            logging.info('Error: response of Llama is neither yes nor no.')
            return -1

    def format_instruction(self, samples):
        formatted_samples = []
        if 'Manipulative' in samples.keys():
            answer_column = 'Manipulative'
            logging.info('Using \"Manipulative\" column for the answers.')
        elif 'Hate Speech' in samples.keys():
            answer_column = 'Hate Speech'
            logging.info('Using \"Hate Speech\" column for the answers.')
        elif 'Condescension' in samples.keys():
            answer_column = 'Condescension'
            logging.info('Using \"Condescension\" column for the answers.')
        elif 'Malevolence' in samples.keys():
            answer_column = 'Malevolence'
            logging.info('Using \"Malevolence\" column for the answers.')
        elif 'Offense' in samples.keys():
            answer_column = 'Offense'
            logging.info('Using \"Offense\" column for the answers.')
        elif 'Stress' in samples.keys():
            answer_column = 'Stress'
            logging.info('Using \"Stress\" column for the answers.')
        elif 'Suicide' in samples.keys():
            answer_column = 'Suicide'
            logging.info('Using \"Suicide\" column for the answers.')
        elif 'Delicacy' in samples.keys():
            answer_column = 'Delicacy'
            logging.info('Using \"Delicacy\" column for the answers.')
        elif 'ToxiGen' in samples.keys():
            answer_column = 'ToxiGen'
            logging.info('Using \"ToxiGen\" column for the answers.')
        else:
            answer_column = 'Manipulative'
            logging.info('Error: the column name of answers is not found. Using \"Manipulative\" as default.')

        for i in range(len(samples['Dialogue'])):
            INSTRUCTION = "### Instruction:\n"
            if answer_column == 'Manipulative':
                INSTRUCTION += """I will provide you with a dialogue. Please determine if it contains elements of mental manipulation. Just answer with 'Yes' or 'No', and don't add anything else.\n\n"""
            elif answer_column == 'Hate Speech':
                INSTRUCTION += """I will provide you with a dialogue. Please determine if it contains elements of hate speech. Just answer with 'Yes' or 'No', and don't add anything else.\n\n"""
            elif answer_column == 'Condescension':
                INSTRUCTION += """I will provide you with a dialogue. Please determine if it contains elements of condescension. Just answer with 'Yes' or 'No', and don't add anything else.\n\n"""
            elif answer_column == 'Malevolence':
                INSTRUCTION += """I will provide you with a dialogue. Please determine if it contains elements of malevolence. Just answer with 'Yes' or 'No', and don't add anything else.\n\n"""
            elif answer_column == 'Offense':
                INSTRUCTION += """I will provide you with a dialogue. Please determine if it contains elements of offense. Just answer with 'Yes' or 'No', and don't add anything else.\n\n"""
            elif answer_column == 'Stress':
                INSTRUCTION += """I will provide you with a piece of text. Please determine if it contains elements of mental stress. Just answer with 'Yes' or 'No', and don't add anything else.\n\n"""
            elif answer_column == 'Suicide':
                INSTRUCTION += """I will provide you with a piece of text. Please determine if it contains suicidal ideation. Just answer with 'Yes' or 'No', and don't add anything else.\n\n"""
            elif answer_column == 'Delicacy':
                INSTRUCTION += """I will provide you with a piece of text. Please determine if it contains delicate toxicity. Just answer with 'Yes' or 'No', and don't add anything else.\n\n"""
            elif answer_column == 'ToxiGen':
                INSTRUCTION += """I will provide you with a piece of text. Please determine if it contains hate speech. Just answer with 'Yes' or 'No', and don't add anything else.\n\n"""
            else:
                INSTRUCTION += """I will provide you with a dialogue. Please determine if it contains elements of mental manipulation. Just answer with 'Yes' or 'No', and don't add anything else.\n\n"""

            INSTRUCTION += "### Dialogue: \n"
            INSTRUCTION += samples['Dialogue'][i]
            INSTRUCTION += "\n\n"
            RESPONSE_KEY = "### Answer: \n"

            if samples[answer_column][i] == '1' or samples[answer_column][i] == 1:
                RESPONSE = "Yes"
            else:
                RESPONSE = "No"
            RESPONSE += "."
            formatted_sample = INSTRUCTION + RESPONSE_KEY + RESPONSE
            formatted_samples.append(formatted_sample)
            # print(formatted_sample)
        return formatted_samples

    def finetuning(self, train_data, valid_data, test_data, epochs, train_batch_size, lr):
        train_data = Dataset.from_pandas(train_data)
        valid_data = Dataset.from_pandas(valid_data)
        test_data = Dataset.from_pandas(test_data)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token  # </s>
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id,
                                                          quantization_config=bnb_config,
                                                          trust_remote_code=True,
                                                          use_cache=False,
                                                          device_map="auto")
        self.model.config.pretraining_tp = 1

        # add "### Answer: \n" to tokenizer
        initial_token_count = len(self.tokenizer)
        instruction_template = "### Instruction:\n"
        dialogue_template = "### Dialogue: \n"
        response_template = "### Answer: \n"
        added_token_count = self.tokenizer.add_special_tokens({"additional_special_tokens": [instruction_template,
                                                                                             dialogue_template,
                                                                                             response_template]})
        self.model.resize_token_embeddings(new_num_tokens=initial_token_count + added_token_count)

        peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='none',
            task_type="CAUSAL_LM",
        )

        model = prepare_model_for_int8_training(self.model)
        model = get_peft_model(model, peft_config)

        collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,
                                                   response_template=response_template,
                                                   tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=self.ft_output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=2,
            optim="paged_adamw_32bit",
            logging_steps=10,
            log_level="warning",
            logging_dir=f"logs",
            logging_strategy="steps",
            learning_rate=lr,
            fp16=True,
            tf32=False,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            disable_tqdm=False,
            save_strategy="steps",
            save_total_limit=1
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=self.tokenizer,
            train_dataset=train_data,
            max_seq_length=4096,
            args=training_args,
            peft_config=peft_config,
            formatting_func=self.format_instruction,
            data_collator=collator,
            callbacks=[LoggingCallback],
        )

        trainer.train()

        # trainer.save_model(self.ft_output_dir)
        model.save_pretrained(self.ft_output_dir)
        self.tokenizer.save_pretrained(self.ft_output_dir)
        logging.info(f"Fine-tuned {self.model_id} saved to {self.ft_output_dir}!")
