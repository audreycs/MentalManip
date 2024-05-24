import os
from datasets import Dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
import logging

from transformers import logging as hf_logging
# Set the logging level for Transformers
hf_logging.set_verbosity_error()
logging.getLogger("transformers.trainer").setLevel(logging.ERROR)


class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logging.info(f"Epoch: {int(state.epoch)}")
            logging.info(f"Best checkpoint: {state.best_model_checkpoint}, Best f1: {state.best_metric}")
            logging.info(logs)


class RoBERTaModel:
    def __init__(self,
                 model,
                 max_length,
                 train_batch_size,
                 valid_batch_size,
                 epochs,
                 learning_rate,
                 output_dir):
        self.model_id = model
        self.max_length = max_length
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_id)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_id,
                                                          do_lower_case=True,
                                                          max_length=self.max_length,  # input token size
                                                          device_map="auto")
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
        macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        accuracy = accuracy_score(labels, predictions)
        conf_matrix = confusion_matrix(labels, predictions)
        return {"precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "f1": micro_f1,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "confusion_matrix": conf_matrix.tolist()}

    def tokenize(self, batch):
        return self.tokenizer(batch["text"], padding=True, truncation=True)

    def finetuning(self, train_data, valid_data, test_data):
        label_column_name = 'Manipulative'
        if 'Hate Speech' in train_data.columns:
            label_column_name = 'Hate Speech'
        elif 'Condescension' in train_data.columns:
            label_column_name = 'Condescension'
        elif 'Malevolence' in train_data.columns:
            label_column_name = 'Malevolence'
        elif 'Offense' in train_data.columns:
            label_column_name = 'Offense'
        elif 'Stress' in train_data.columns:
            label_column_name = 'Stress'
        elif 'Suicide' in train_data.columns:
            label_column_name = 'Suicide'
        elif 'Delicacy' in train_data.columns:
            label_column_name = 'Delicacy'
        elif 'ToxiGen' in train_data.columns:
            label_column_name = 'ToxiGen'
        else:
            label_column_name = 'Manipulative'
        train_data["label"] = train_data[label_column_name].apply(lambda x: 0 if x == '0' else 1)
        valid_data["label"] = valid_data[label_column_name].apply(lambda x: 0 if x == '0' else 1)
        test_data["label"] = test_data['Manipulative'].apply(lambda x: 0 if x == '0' else 1)
        self.train_data = Dataset.from_pandas(train_data).rename_column("Dialogue", "text")
        self.valid_data = Dataset.from_pandas(valid_data).rename_column("Dialogue", "text")
        self.test_data = Dataset.from_pandas(test_data).rename_column("Dialogue", "text")

        train_dataset = self.train_data.map(self.tokenize, batched=True, batch_size=len(self.train_data))
        valid_dataset = self.valid_data.map(self.tokenize, batched=True, batch_size=len(self.valid_data))
        test_dataset = self.test_data.map(self.tokenize, batched=True, batch_size=len(self.test_data))

        # Set dataset format
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        valid_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        if os.path.exists(self.output_dir) is False:
            os.makedirs(self.output_dir)

        # TrainingArguments
        training_args = TrainingArguments(
            # The output directory where the model predictions and checkpoints will be written
            output_dir=self.output_dir,
            optim="adamw_torch",
            # The initial learning rate for AdamW optimizer
            learning_rate=self.learning_rate,
            # Total number of training epochs to perform
            num_train_epochs=self.epochs,
            # The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training
            per_device_train_batch_size=self.train_batch_size,
            # The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation
            per_device_eval_batch_size=self.valid_batch_size,
            # Evaluation is done at the end of each epoch. can be selected to {‘no’, ‘steps’, ‘epoch’}.
            evaluation_strategy="epoch",
            log_level="warning",
            logging_dir=f"logs",
            logging_strategy="no",
            disable_tqdm=False,
            # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.
            weight_decay=0.01,
            # Number of steps used for a linear warmup from 0 to learning_rate
            warmup_steps=500,
            # The checkpoint save strategy to adopt during training
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.
            save_total_limit=1,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[LoggingCallback],
        )

        # Fine-tune the model
        trainer.train()

        logging.info("")
        logging.info("-----Test-----")

        # Evaluate the model
        result = trainer.evaluate(test_dataset)
        for key, value in result.items():
            logging.info(f"{key} = {value}")
