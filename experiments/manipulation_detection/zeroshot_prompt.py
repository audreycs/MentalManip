import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import argparse

from load_data import LoadManipDataset
from model_chatgpt import ChatGPTModel
from model_llama import LlamaModel
from utils import *


def prediction(model, test_data):
    targets = [int(v) for v in test_data['Manipulative'].values]
    preds = []
    count = 0
    for idx, row in test_data.iterrows():
        count += 1
        logging.info(f"-----Running {model.model_id} zeroshot prompting ({count}/{len(test_data)})-----")
        dialogue = row['Dialogue']
        pred = model.zeroshot_prompting(dialogue)
        preds.append(pred)

    corrupted_result = 0
    processed_preds, processed_targets = [], []
    for pred, target in zip(preds, targets):
        if pred == -1:
            corrupted_result += 1
        else:
            processed_preds.append(pred)
            processed_targets.append(target)

    logging.info(f"\n----------{model.model_id} zero prompting result----------")
    logging.info(
        f"Out of {len(preds)} test samples, corrupted samples: {corrupted_result}, processed samples: {len(processed_preds)}")

    # Calculate metrics
    precision = precision_score(processed_targets, processed_preds, zero_division=0)
    recall = recall_score(processed_targets, processed_preds, zero_division=0)
    micro_f1 = f1_score(processed_targets, processed_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(processed_targets, processed_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(processed_targets, processed_preds)
    conf_matrix = confusion_matrix(processed_targets, processed_preds)

    # Print results
    logging.info(f"Golden manipulative samples = {len([v for v in processed_targets if v == 1])}, non-manipulative samples = {len([v for v in processed_targets if v == 0])}")
    logging.info(f"Predicted manipulative samples = {len([v for v in processed_preds if v == 1])}, non-manipulative samples = {len([v for v in processed_preds if v == 0])}")
    logging.info(f"- Precision = {precision:.3f}")
    logging.info(f"- Recall = {recall:.3f}")
    logging.info(f"- Accuracy = {accuracy:.3f}")
    logging.info(f"- Micro F1-Score = {micro_f1:.3f}")
    logging.info(f"- Macro F1-Score = {macro_f1:.3f}")
    logging.info(f"- Confusion Matrix = \n{conf_matrix}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='zeroshot')
    parser.add_argument('--model', default='chatgpt', type=str)
    parser.add_argument('--temp', default=0.1, type=float)
    parser.add_argument('--top_p', default=0.5, type=float)
    parser.add_argument('--penal', default=0.0, type=float)
    parser.add_argument('--log_dir', default='./logs', type=str)
    parser.add_argument('--data', default='../datasets/mentalmanip_maj.csv', type=str)
    args = parser.parse_args()

    if os.path.exists(args.log_dir) is False:
        os.makedirs(args.log_dir)

    set_logging(args, parser.description)
    show_args(args)

    manip_dataset = LoadManipDataset(file_name=args.data,
                                     train_ratio=0.6,
                                     valid_ratio=0.2,
                                     test_ratio=0.2,
                                     split_draw=False)

    test_data = manip_dataset.df_test

    if args.model == 'chatgpt':
        modelChatgpt = ChatGPTModel(gpt_model="gpt-4-1106-preview",
                                    api_key="",  # Please provide your OpenAI API key
                                    temperature=0.1,
                                    top_p=0.5,
                                    penal=0.0,
                                    max_input_token_length=4096)
        prediction(modelChatgpt, test_data)

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

        prediction(modelLlama, test_data)
