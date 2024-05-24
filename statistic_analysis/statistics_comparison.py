import csv
import os
import json
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
from collections import Counter
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import re
import warnings
import statistics
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sentence_transformers import SentenceTransformer
import torch
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Filter out the specific FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning,
                        message='.*is_categorical_dtype is deprecated.*')


def read_file(folder):
    if 'data' in folder:
        # MentalManip
        dialogues = []
        with open(os.path.join(folder, 'new_processed_mentalmanip_maj_final.csv'), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for index, row in enumerate(reader):
                if index == 0:
                    continue
                else:
                    dialogue = row[1].strip()
                    dialogues.append(dialogue)
        return dialogues
    elif 'ToxiChat' in folder:
        dialogues = []
        for file in ['train.jsonl', 'dev.jsonl', 'test.jsonl']:
            with open(os.path.join(folder, file), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    uttrances = []
                    for index, turn in enumerate(data['reddit_thread']):
                        if index == 0:
                            uttrance = ' '.join(turn['text'].split('\n')[1:]).strip()
                        else:
                            uttrance = turn['text'].strip()
                        uttrances.append('utterance ' + str(id) + ': ' + uttrance)
                    dialogues.append('\n'.join(uttrances))
        return dialogues
    elif 'TalkDown' in folder:
        dialogues = []
        with open(os.path.join(folder, 'talkdown-dialogues.csv'), 'r') as f:
            reader = csv.reader(f)
            for index, row in enumerate(reader):
                dialogue = row[0].strip()
                dialogues.append(dialogue)
        return dialogues
    elif 'MDMD' in folder:
        df = pd.read_csv(os.path.join(folder, 'MDMD_dataset.tsv'), sep='\t', header=0, encoding='utf-8')
        # remove '@USER' in the text in column 'utterance'
        df['utterance'] = df['utterance'].apply(lambda x: x.replace('@USER', ''))
        df['utterance'] = df['utterance'].apply(lambda x: x.strip())
        # add 'utterance' together with value in column 'Turn' in front of each text in column 'utterance'
        df['utterance'] = 'Turn ' + df['Turn'].astype(str) + ": " + df['utterance']
        # concatenate the utterances in the same dialogue
        concatenated_df = df.groupby('Number')['utterance'].apply('\n'.join).reset_index()
        dialogues = concatenated_df['utterance'].tolist()
        return dialogues
    elif 'fox-news' in folder:
        dialogues = []
        with open(os.path.join(folder, 'dataset.csv'), 'r') as f:
            reader = csv.reader(f)
            for index, row in enumerate(reader):
                dialogue = row[0].strip()
                dialogues.append(dialogue)
        return dialogues
    else:
        print("Please input a valid folder name!")
        exit(1)


def token_size(dialogues, flag='labeled'):
    token_size = []
    if 'labeled'.lower() == flag.lower():
        for dialogue in dialogues:
            content = dialogue.replace('Person1: ', '').replace('Person2: ', '')
            token_size.append(len(word_tokenize(content)))
    elif 'TalkDown'.lower() == flag.lower():
        for dialogue in dialogues:
            content = dialogue.replace('Person1: ', '').replace('Person2: ', '')
            token_size.append(len(word_tokenize(content)))
    elif 'ToxiChat'.lower() == flag.lower():
        for dialogue in dialogues:
            content = dialogue.replace('utterance 0: ', '').replace('utterance 1: ', '')
            token_size.append(len(word_tokenize(content)))
    elif 'MDMD'.lower() == flag.lower():
        for dialogue in dialogues:
            content = dialogue.replace('Turn 0: ', '').replace('Turn 1: ', '')
            token_size.append(len(word_tokenize(content)))
    elif 'fox-news'.lower() == flag.lower():
        for dialogue in dialogues:
            token_size.append(len(word_tokenize(dialogue)))
    return token_size


def turn_num(dialogues, flag=''):
    if flag == 'TalkDown':
        return [2]*len(dialogues), 2, 0
    turn_numbers = []
    for dialogue in dialogues:
        turns = dialogue.split('\n')
        turn_numbers.append(len(turns))
    return turn_numbers, np.mean(turn_numbers), statistics.stdev(turn_numbers)


def ccdf(data_list, file_name, x_lim, title):
    data_list = [sorted(data, reverse=False) for data in data_list]
    our_turns, toxichat_turns, talkdown_turns, mdmd_turns, foxnews_turns = data_list

    # Calculate the CDF
    our_turns_cdf = np.arange(len(our_turns)) / float(len(our_turns))
    toxichat_turns_cdf = np.arange(len(toxichat_turns)) / float(len(toxichat_turns))
    talkdown_turns_cdf = np.arange(len(talkdown_turns)) / float(len(talkdown_turns))
    mdmd_turns_cdf = np.arange(len(mdmd_turns)) / float(len(mdmd_turns))
    foxnews_turns_cdf = np.arange(len(foxnews_turns)) / float(len(foxnews_turns))

    # Calculate the CCDF
    our_turns_ccdf = 1 - our_turns_cdf
    toxichat_turns_ccdf = 1 - toxichat_turns_cdf
    talkdown_turns_ccdf = 1 - talkdown_turns_cdf
    mdmd_turns_ccdf = 1 - mdmd_turns_cdf
    foxnews_turns_ccdf = 1 - foxnews_turns_cdf

    # Set the Seaborn style
    sns.set_theme()
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    plt.figure(figsize=(13, 8))
    plt.tight_layout()
    # Plotting the CCDF
    linewidth_ = 3
    plt.plot(foxnews_turns, foxnews_turns_ccdf, label='Fox News', linewidth=linewidth_)
    plt.plot(talkdown_turns, talkdown_turns_ccdf, label='TalkDown', linewidth=linewidth_)
    plt.plot(toxichat_turns, toxichat_turns_ccdf, label='ToxiChat', linewidth=linewidth_)
    plt.plot(mdmd_turns, mdmd_turns_ccdf, label='MDRDC', linewidth=linewidth_)
    plt.plot(our_turns, our_turns_ccdf, label='MentalManip', linewidth=linewidth_)

    # Function to format y-axis as percentage
    def to_percentage(x, pos):
        return f'{100 * x:.0f}%'

    # Format y-axis as percentage
    formatter = FuncFormatter(to_percentage)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Get the current axes
    ax = plt.gca()

    # Set the color of the spines (borders) to black
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    plt.xlim(0, x_lim)
    plt.xlabel(title, fontsize=30)
    plt.ylabel('Percentage', fontsize=30)
    plt.legend(loc='upper right', fontsize=25)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    # Draw horizontal lines for percentiles (e.g., 25th, 50th, 75th)
    # for percentile in [0.25, 0.5, 0.75]:
    #     plt.axhline(y=1 - percentile, color='r', linestyle='--')

    # plt.title('CCDF Comparison of Utterance Number Distribution', fontsize=30)
    plt.savefig(file_name, format='pdf', bbox_inches='tight')


def sentiment_analysis(dialogues):
    tokenizer = AutoTokenizer.from_pretrained('bhadresh-savani/distilbert-base-uncased-emotion')
    classifier = pipeline("text-classification",
                          model='bhadresh-savani/distilbert-base-uncased-emotion',
                          top_k=True)
    scores = []
    for dialogue in tqdm(dialogues):
        dialogue = dialogue.replace('Person1: ', '').replace('Person2: ', '')
        dialogue = dialogue.replace('utterance 0: ', '').replace('utterance 1: ', '')
        dialogue = dialogue.replace('Turn 0: ', '').replace('Turn 1: ', '')
        tokens = tokenizer.tokenize(dialogue)
        truncated_tokens = tokens[:500]
        truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)

        prediction = classifier(truncated_text)
        scores.append(prediction[0][0]['label'])
    return Counter(scores)


def preprocessing(dialogues):
    processed_dialogues = []
    for dialogue in dialogues:
        content = re.sub(r'Person\d+:', '<person>:', dialogue)
        content = re.sub(r'Turn \d+:', '<person>:', content)
        content = re.sub(r'utterance \d+:', '<person>:', content)
        if '<person>:' not in content:
            content = '\n'.join(['<person>: '+i for i in content.split('\n')])
        processed_dialogues.append(content)
    return processed_dialogues


def embedding_space(dialogues_list):
    # Load pre-trained model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    df_list = []
    df_name_list = ['MentalManip', 'TalkDown', 'ToxiChat', 'MDRDC', 'Fox News']

    for dialogues in dialogues_list:
        processed_dialogues = preprocessing(dialogues)
        embeddings = model.encode(processed_dialogues)

        # Reduce dimensions using t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=0)
        two_dim_embeddings = tsne.fit_transform(np.array(embeddings))

        # Prepare DataFrame for Seaborn
        df_list.append(pd.DataFrame(two_dim_embeddings, columns=['x', 'y']))
        df_list[-1]['dataset'] = df_name_list[len(df_list)-1]

        np.savetxt(f'embeddings_{df_name_list[len(df_list)-1]}.csv', two_dim_embeddings, delimiter=',')

    df = pd.concat(df_list)
    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(data=df, x='x', y='y', hue='dataset',
                         palette='colorblind', style='dataset',
                         s=80)
    ax.set_xlabel('X', fontsize=25)
    ax.set_ylabel('Y', fontsize=25)
    ax.xaxis.set_label_coords(0.9, 0.05)
    ax.yaxis.set_label_coords(-0.01, 1.0)
    ax.tick_params(axis='x', labelsize=23)
    ax.tick_params(axis='y', labelsize=23)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    plt.legend(fontsize=20, ncol=2, loc='upper left')
    plt.grid(False)
    # plt.title('SentenceTransformer Embeddings after t-SNE', fontsize=20)
    # Save the figure with high resolution
    plt.savefig('word_embedding_space_comparison.png', dpi=500, bbox_inches='tight')
    plt.savefig('embedding_space_comparison.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    our_dialogues = read_file(folder='./data')
    talkdown_dialogues = read_file(folder='./OtherDatasets/TalkDown2019')
    toxichat_dialogues = read_file(folder='./OtherDatasets/ToxiChat2021')
    mdmd_dialogues = read_file(folder='./OtherDatasets/MDMD2021')
    foxnews_dialogeus = read_file(folder='./OtherDatasets/fox-news2017')
    print(f"Dialogue number (our_dialogues): {len(our_dialogues)}")
    print(f"Dialogue number (talkdown_dialogues): {len(talkdown_dialogues)}")
    print(f"Dialogue number (toxichat_dialogues): {len(toxichat_dialogues)}")
    print(f"Dialogue number (mdmd_dialogues): {len(mdmd_dialogues)}")
    print(f"Dialogue number (foxnews_dialogeus): {len(foxnews_dialogeus)}")

    # get token size information
    print("----------Token Size----------")
    our_token_size = token_size(our_dialogues, flag='labeled')
    talkdown_token_size = token_size(talkdown_dialogues, flag='TalkDown')
    toxichat_token_size = token_size(toxichat_dialogues, flag='ToxiChat')
    mdmd_token_size = token_size(mdmd_dialogues, flag='MDMD')
    foxnews_token_size = token_size(foxnews_dialogeus, flag='fox-news')
    print("Our token size: ", np.mean(our_token_size))
    print("TalkDown token size: ", np.mean(talkdown_token_size))
    print("ToxiChat token size: ", np.mean(toxichat_token_size))
    print("MDMD token size: ", np.mean(mdmd_token_size))
    print("Fox-news token size: ", np.mean(foxnews_token_size))

    # get the number of turns
    print("----------Turn Number----------")
    our_turns, our_turn_avg, our_turn_var = turn_num(our_dialogues)
    toxichat_turns, toxichat_turn_avg, toxichat_turn_var = turn_num(toxichat_dialogues)
    talkdown_turns, talkdown_turn_avg, talkdown_turn_var = turn_num(talkdown_dialogues, flag='TalkDown')
    mdmd_turns, mdmd_turn_avg, mdmd_turn_var = turn_num(mdmd_dialogues)
    foxnews_turns, foxnews_turn_avg, foxnews_turn_var = turn_num(foxnews_dialogeus)
    print(f"Our turn number: {our_turn_avg}, var: {our_turn_var}")
    print(f"TalkDown turn number: {talkdown_turn_avg}, var: {talkdown_turn_var}")
    print(f"ToxiChat turn number: {toxichat_turn_avg}, var: {toxichat_turn_var}")
    print(f"MDMD turn number: {mdmd_turn_avg}, var: {mdmd_turn_var}")
    print(f"Fox-news turn number: {foxnews_turn_avg}, var: {foxnews_turn_var}")

    # draw ccdf of utterance number distribution
    # ccdf([our_turns, toxichat_turns, talkdown_turns, mdmd_turns, foxnews_turns],
    #      file_name="draw_ccdf_utterance.pdf", x_lim=20, title="Utterance Number")
    # # draw ccdf of token number distribution
    # ccdf([our_token_size, toxichat_token_size, talkdown_token_size, mdmd_token_size, foxnews_token_size],
    #      file_name="draw_ccdf_token.pdf", x_lim=2000, title="Token Number")

    # get sentiment scores
    # print("----------Sentiment Scores----------")
    # our_sentiment_scores = sentiment_analysis(our_dialogues)
    # toxichat_sentiment_scores = sentiment_analysis(toxichat_dialogues)
    # mdmd_sentiment_scores = sentiment_analysis(mdmd_dialogues)
    # talkdown_sentiment_scores = sentiment_analysis(talkdown_dialogues)
    # foxnews_sentiment_scores = sentiment_analysis(foxnews_dialogeus)
    # print(our_sentiment_scores)
    # print(talkdown_sentiment_scores)
    # print(toxichat_sentiment_scores)
    # print(mdmd_sentiment_scores)
    # print(foxnews_sentiment_scores)
    # with open("sentiment_result.json", "w") as file:
    #     for result in [our_sentiment_scores, talkdown_sentiment_scores, toxichat_sentiment_scores, mdmd_sentiment_scores, foxnews_sentiment_scores]:
    #         json.dump(result, file)
    #         file.write('\n')

    # get embedding space
    # print("----------Embedding Space----------")
    # size = 500
    # np.random.seed(11)
    # selected_our_dialogues = np.random.choice(our_dialogues, size, replace=False)
    # selected_talkdown_dialogues = np.random.choice(talkdown_dialogues, size, replace=False)
    # selected_toxichat_dialogues = np.random.choice(toxichat_dialogues, size, replace=False)
    # selected_mdmd_dialogues = np.random.choice(mdmd_dialogues, size, replace=False)
    # selected_foxnews_dialogues = np.random.choice(foxnews_dialogeus, size, replace=False)
    #
    # embedding_space([selected_our_dialogues,
    #                 selected_talkdown_dialogues,
    #                 selected_toxichat_dialogues,
    #                 selected_mdmd_dialogues,
    #                 selected_foxnews_dialogues])
    # print(f"embedding space plotting finished!")
