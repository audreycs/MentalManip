import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import argparse
import numpy as np

from load_data import LoadManipDataset
from model_chatgpt import ChatGPTModel
from model_llama import LlamaModel
from utils import *


def prediction(model, test_data, manip_examples, nonmanip_examples):
    targets = [int(v) for v in test_data['Manipulative'].values]
    preds = []
    count = 0
    for idx, row in test_data.iterrows():
        count += 1
        logging.info(f"-----Running {model.model_id} fewshot prompting ({count}/{len(test_data)})-----")
        dialogue = row['Dialogue']
        pred = model.fewshot_prompting(manip_examples, nonmanip_examples, dialogue)
        preds.append(pred)

    corrupted_result = 0
    processed_preds, processed_targets = [], []
    for pred, target in zip(preds, targets):
        if pred == -1:
            corrupted_result += 1
        else:
            processed_preds.append(pred)
            processed_targets.append(target)

    logging.info(f"\n----------{model.model_id} fewshot prompting result----------")
    logging.info(f"Out of {len(preds)} test samples, corrupted samples: {corrupted_result}, processed samples: {len(processed_preds)}")

    # Calculate metrics
    precision = precision_score(processed_targets, processed_preds, zero_division=0)
    recall = recall_score(processed_targets, processed_preds, zero_division=0)
    micro_f1 = f1_score(processed_targets, processed_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(processed_targets, processed_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(processed_targets, processed_preds)
    conf_matrix = confusion_matrix(processed_targets, processed_preds)

    # Print results
    logging.info(
        f"Golden manipulative samples = {len([v for v in processed_targets if v == 1])}, non-manipulative samples = {len([v for v in processed_targets if v == 0])}")
    logging.info(
        f"Predicted manipulative samples = {len([v for v in processed_preds if v == 1])}, non-manipulative samples = {len([v for v in processed_preds if v == 0])}")
    logging.info(f"- Precision = {precision:.3f}")
    logging.info(f"- Recall = {recall:.3f}")
    logging.info(f"- Accuracy = {accuracy:.3f}")
    logging.info(f"- Micro F1-Score = {micro_f1:.3f}")
    logging.info(f"- Macro F1-Score = {macro_f1:.3f}")
    logging.info(f"- Confusion Matrix = \n{conf_matrix}")


def select_examples(data, k_manip=1, k_nonmanip=2):
    manip_list = data.df_train[data.df_train['Manipulative'] == '1'].index.tolist()
    # randomly select k_manip examples from manip_list
    selected_manip_idx = np.random.choice(manip_list, k_manip, replace=False)
    nonmanip_list = data.df_train[data.df_train['Manipulative'] == '0'].index.tolist()
    # randomly select k_nonmanip examples from nonmanip_list
    selected_nonmanip_idx = np.random.choice(nonmanip_list, k_nonmanip, replace=False)
    manip_examples = data.df_train.loc[selected_manip_idx]
    nonmanip_examples = data.df_train.loc[selected_nonmanip_idx]
    return manip_examples, nonmanip_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fewshot')
    parser.add_argument('--model', default='llama-13b', type=str)
    parser.add_argument('--temp', default=0.1, type=float)
    parser.add_argument('--top_p', default=0.5, type=float)
    parser.add_argument('--penal', default=0.0, type=float)
    parser.add_argument('--num_manip', default=2, type=int)
    parser.add_argument('--num_nonmanip', default=1, type=int)
    parser.add_argument('--log_dir', default='./logs', type=str)
    parser.add_argument('--data', default='../datasets/mentalmanip_con.csv', type=str)
    args = parser.parse_args()

    if os.path.exists(args.log_dir) is False:
        os.makedirs(args.log_dir)

    set_logging(args, parser.description)
    show_args(args)

    manip_dataset = LoadManipDataset(file_name=args.data,
                                     train_ratio=0.6,
                                     valid_ratio=0.2,
                                     test_ratio=0.2)

    manip_examples, nonmanip_examples = select_examples(manip_dataset, k_manip=args.num_manip, k_nonmanip=args.num_nonmanip)
    test_data = manip_dataset.df_test

    if args.model == 'chatgpt':
        modelChatgpt = ChatGPTModel(gpt_model="gpt-4-1106-preview",
                                    api_key="",  # add your OpenAI API key
                                    temperature=0.1,
                                    top_p=0.5,
                                    penal=0.0,
                                    max_input_token_length=4096)
        prediction(modelChatgpt, test_data, manip_examples, nonmanip_examples)
    elif 'llama' in args.model:
        llama_model = "Llama-2-7b-chat-hf"
        if '13b' in args.model:
            llama_model = "Llama-2-13b-chat-hf"
        modelLlama = LlamaModel(load_from_local=False,
                                model=llama_model,
                                temperature=0.6,
                                top_p=0.9,
                                top_k=50,
                                repetition_penalty=1.2,
                                max_new_tokens=1024,
                                max_input_token_length=4096,
                                ft_output_dir='llama_ft')
        prediction(modelLlama, test_data, manip_examples, nonmanip_examples)
