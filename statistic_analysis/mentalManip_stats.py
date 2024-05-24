import csv
import numpy as np
import seaborn as sns
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import torch

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
import re
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def preprocessing(dialogues):
    processed_dialogues = []
    for dialogue in dialogues:
        content = re.sub(r'Person\d+:', '<person>:', dialogue)
        content = re.sub(r'Turn \d+:', '<person>:', content)
        content = re.sub(r'utterance \d+:', '<person>:', content)
        if '<person>:' not in content:
            content = '\n'.join(['<person>: ' + i for i in content.split('\n')])
        processed_dialogues.append(content)
    return processed_dialogues


def embedding_space(dialogues_list):
    # Load pre-trained model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

    manip_dialogues, nonmanip_dialogues = dialogues_list[0], dialogues_list[1]
    select_size = min(len(manip_dialogues), len(nonmanip_dialogues))

    df_list = []
    df_name_list = ['Manipulative', 'Non-manipulative']
    scaler = StandardScaler()
    for dialogues in dialogues_list:
        selected_dialogues = np.random.choice(dialogues, size=select_size, replace=False)
        processed_dialogues = preprocessing(selected_dialogues)
        embeddings = model.encode(processed_dialogues)
        standardized_embeddings = scaler.fit_transform(embeddings)

        # Reduce dimensions using t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=0)
        two_dim_embeddings = tsne.fit_transform(np.array(embeddings))

        # Prepare DataFrame for Seaborn
        df_list.append(pd.DataFrame(two_dim_embeddings, columns=['x', 'y']))
        df_list[-1]['dataset'] = df_name_list[len(df_list) - 1]
    df = pd.concat(df_list)

    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(data=df, x='x', y='y', hue='dataset',
                         palette=['blue', 'orange'], style='dataset', s=80)
    ax.set_xlabel('X', fontsize=25)
    ax.set_ylabel('Y', fontsize=25)
    ax.xaxis.set_label_coords(0.9, 0.05)
    ax.yaxis.set_label_coords(-0.01, 1.0)
    ax.tick_params(axis='x', labelsize=23)
    ax.tick_params(axis='y', labelsize=23)
    ax.grid(False)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    plt.legend(fontsize=18, ncol=1, loc='upper left')
    # plt.title('Embedding Space of Manip vs. Non-manip', fontsize=20)
    # Save the figure with high resolution
    plt.savefig('draw_(non)manip_space_comparison.pdf', format='pdf', bbox_inches='tight')


def draw_plot(df, figsize, title, save_file_name, y_height, font_size):
    # Create a new figure for the plot
    plt.figure(figsize=figsize)

    # Apply the default theme
    sns.set_theme()
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    sns.color_palette("tab10")
    ax = sns.barplot(data=df,
                     x="label",
                     y="count",
                     hue="Dataset",
                     width=0.9,
                     palette="tab10")
    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    print(x_labels)

    if 'technique' in title.lower():
        x_tick_label = ['P_S', 'ACC', 'INT', 'S_B', 'RAT', 'DEN', 'EVA', 'FEI', 'B_A', 'VIC', 'SER']
    else:
        x_tick_label = ['O_R', 'NAT', 'DEP', 'L_S', 'O_I']
    ax.set_xticklabels(x_tick_label)

    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2.,  # x position
                p.get_height(),  # y position
                '{:.0f}'.format(p.get_height()),  # label
                ha='center',  # horizontal alignment
                va='bottom',  # vertical alignment
                color='black',  # text color
                fontsize=22,
                )

    ax.tick_params(axis='y', labelsize=24)  # Adjust label size as needed
    ax.tick_params(axis='x', labelbottom=True, labelsize=24)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(title, fontsize=28)
    ax.set_ylim(0, y_height)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    # set legend size
    ax.legend(fontsize=24)
    plt.savefig(save_file_name, format='pdf', bbox_inches='tight')
    plt.show()


def sentiment_analysis(dialogues, manip_list):
    manip_list = ['Manipulative' if i == 1 or i == '1' else 'Non-manipulative' for i in manip_list]
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

    # create a dataframe
    senti_df = pd.DataFrame({'dialogue': dialogues, 'senti': scores, 'manip': manip_list})

    return senti_df


def draw_sentiment_plot(df, title, save_file_name, y_height, font_size):
    # Create a new figure for the plot
    plt.figure()

    sns.set_theme()
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    desired_order = ['anger', 'joy', 'sadness', 'fear', 'love', 'surprise']
    df['senti'] = pd.Categorical(df['senti'], categories=desired_order, ordered=True)
    ax = sns.histplot(data=df,
                      x="senti",
                      hue="manip",
                      multiple="dodge",
                      shrink=.9,
                      palette=["skyblue", "orange"],
                      alpha=1.0)
    ax.grid(axis='x')
    ax.set_ylim(0, y_height)
    ax.tick_params(axis='y', labelsize=16)  # Adjust label size as needed
    ax.tick_params(axis='x', labelsize=16)
    ax.legend_.set_title(None)
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    ax.set_title(title, fontsize=16)
    for idx, p in enumerate(ax.patches):
        if idx == 4 or idx == 10:
            ax.text(p.get_x() + p.get_width() / 2.,  # x position
                    p.get_height(),  # y position
                    '{:.0f}'.format(p.get_height()),  # label
                    ha='center',  # horizontal alignment
                    va='bottom',  # vertical alignment
                    color='black',  # text color
                    fontsize=14,
                    )
        else:
            ax.text(p.get_x() + p.get_width() / 2.,  # x position
                    p.get_height(),  # y position
                    '{:.0f}'.format(p.get_height()),  # label
                    ha='center',  # horizontal alignment
                    va='bottom',  # vertical alignment
                    color='black',  # text color
                    fontsize=16,
                    )
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    plt.setp(ax.get_legend().get_texts(), fontsize='16')
    plt.savefig(save_file_name, format='pdf', bbox_inches='tight')


def draw_heatmap(value_list, flag):
    tech_columns = ["Denial",
                    "Rationalization",
                    "Feigning Innocence",
                    "Evasion",
                    "Intimidation",
                    "Shaming or Belittlement",
                    "Accusation",
                    "Playing Victim Role",
                    "Playing Servant Role",
                    "Persuasion or Seduction",
                    "Brandishing Anger"]
    short_techs = ["DEN", "RAT", "F_I", "EVA", "INT", "S_B", "ACC", "VIC", "SER", "P_S", "B_A"]
    tech_pos_dict = {c: idx for idx, c in enumerate(tech_columns)}
    vul_columns = ["Over-responsibility", "Over-intellectualization", "Naivete", "Low self-esteem", "Dependency"]
    short_vuls = ["O_R", "O_I", "NAT", "L_S", "DEP"]
    vul_pos_dict = {c: idx for idx, c in enumerate(vul_columns)}
    if flag == 'tech':
        df = pd.DataFrame(columns=tech_columns)
        for row in value_list:
            if row != "":
                row_list = [0] * len(tech_columns)
                techs = row.split(',')
                for t in techs:
                    row_list[tech_pos_dict[t]] = 1
                df.loc[len(df)] = row_list
        df = df.astype(int)
        co_occurrence_matrix = np.dot(df.transpose(), df)
        # co_occurrence_matrix = co_occurrence_matrix / np.max(co_occurrence_matrix)
        np.fill_diagonal(co_occurrence_matrix, 0)
        co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=short_techs, columns=short_techs)
        # make diagonal elements masked
        mask = np.eye(co_occurrence_df.shape[0], dtype=bool)
    elif flag == 'vul':
        df = pd.DataFrame(columns=vul_columns)
        for row in value_list:
            if row != "":
                row_list = [0] * len(vul_columns)
                vuls = row.split(',')
                for v in vuls:
                    row_list[vul_pos_dict[v]] = 1
                df.loc[len(df)] = row_list
        df = df.astype(int)
        co_occurrence_matrix = np.dot(df.transpose(), df)
        # co_occurrence_matrix = co_occurrence_matrix / np.max(co_occurrence_matrix)
        np.fill_diagonal(co_occurrence_matrix, 0)
        co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=short_vuls, columns=short_vuls)
        # make diagonal elements masked
        mask = np.eye(co_occurrence_df.shape[0], dtype=bool)

    elif flag == 'tech_vul':
        tech_list, vul_list = value_list
        df = pd.DataFrame(columns=vul_columns)
        for idx, c in enumerate(tech_columns):
            df.loc[c] = [0] * len(vul_columns)
        df = df.astype(int)
        for tech_row, vul_row in zip(tech_list, vul_list):
            if tech_row != "" and vul_row != "":
                techs = tech_row.split(',')
                vuls = vul_row.split(',')
                for t in techs:
                    for v in vuls:
                        df.loc[t, v] += 1
        co_occurrence_matrix = df.values
        # co_occurrence_matrix = co_occurrence_matrix / np.max(co_occurrence_matrix)
        co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=short_techs, columns=short_vuls)
        mask = None
    else:
        raise ValueError("Invalid flag!")

    plt.figure(figsize=(10, 8))
    sns.set_theme()
    ax = sns.heatmap(co_occurrence_df,
                     annot=False,  # Annotate cells with their values
                     cmap="YlGnBu",  # Color map
                     linewidths=.5,  # Space between cells
                     annot_kws={"size": 20},
                     mask=mask)  # Adjust annotation font size
    # plt.title("Label Co-occurrence Heatmap")
    ax.tick_params(axis='y', labelsize=25)  # Adjust label size as needed
    ax.tick_params(axis='x', labelsize=25)
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    # not showing xticks
    # if flag != 'tech_vul':
    #     plt.xticks([])
    # making xticks rotate
    if flag != 'tech_vul' and flag != 'vul':
        plt.xticks(rotation=40)
    plt.tight_layout()
    # Show the plot
    plt.savefig(f"draw_heatmap_{flag}_maj.pdf", format='pdf', bbox_inches='tight')


def get_stats(file):
    dialogue_list = []
    manip_list = []
    tech_list = []
    vul_list = []
    manip_dialogues = []
    nonmanip_dialogues = []
    with open(file, 'r', newline='', encoding='utf-8') as infile:
        content = csv.reader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for index, row in enumerate(content):
            if index == 0:
                continue
            id, dialogue, manip, tech, vul = row[:5]
            dialogue_list.append(dialogue)
            manip_list.append(manip)
            tech_list.append(tech)
            vul_list.append(vul)
            if manip == "1":
                manip_dialogues.append(dialogue)
            else:
                nonmanip_dialogues.append(dialogue)

    indi_tech_list = []
    for t in tech_list:
        if t != "":
            indi_tech_list.extend(t.split(','))
    tech_dic = Counter(indi_tech_list)
    tech_df = pd.DataFrame.from_dict(tech_dic, orient='index').reset_index()
    tech_df.columns = ['label', 'count']
    tech_df = tech_df.sort_values(by='count', ascending=False)

    indi_vul_list = []
    for v in vul_list:
        if v != "":
            indi_vul_list.extend(v.split(','))
    vul_dic = Counter(indi_vul_list)
    vul_df = pd.DataFrame.from_dict(vul_dic, orient='index').reset_index()
    vul_df.columns = ['label', 'count']
    vul_df = vul_df.sort_values(by='count', ascending=False)

    return dialogue_list, manip_list, tech_list, vul_list, manip_dialogues, nonmanip_dialogues, tech_df, vul_df


if __name__ == '__main__':
    file1 = "../mentalmanip_dataset/mentalmanip_con.csv"
    file2 = "../mentalmanip_dataset/mentalmanip_maj.csv"
    (dialogue_list_1, manip_list_1, tech_list_1, vul_list_1, manip_dialogues_1,
     nonmanip_dialogues_1, tech_df1, vul_df1) = get_stats(file1)
    (dialogue_list_2, manip_list_2, tech_list_2, vul_list_2, manip_dialogues_2,
     nonmanip_dialogues_2, tech_df2, vul_df2) = get_stats(file2)
    tech_df1['Dataset'] = 'MentalManip_con'
    tech_df2['Dataset'] = 'MentalManip_maj'
    vul_df1['Dataset'] = 'MentalManip_con'
    vul_df2['Dataset'] = 'MentalManip_maj'

    # merge two dataframes without index
    tech_df = pd.concat([tech_df1, tech_df2], ignore_index=True)
    vul_df = pd.concat([vul_df1, vul_df2], ignore_index=True)

    # draw distribution of techniques and vulnerabilities
    draw_plot(tech_df, (12,8), "Count of Different Techniques", "draw_techniques.pdf",
              y_height=800, font_size=14)
    draw_plot(vul_df, (8,8), "Count of Different Vulnerabilities", "draw_vulnerabilities.pdf",
              y_height=400, font_size=14)

    # draw distribution of sentiment scores
    senti_df_1 = sentiment_analysis(dialogue_list_1, manip_list_1)
    senti_df_2 = sentiment_analysis(dialogue_list_2, manip_list_2)
    draw_sentiment_plot(senti_df_1, "Emotion Distribution of MentalManip_con", "draw_senti_scores_con.pdf",
                        y_height=750, font_size=12)
    draw_sentiment_plot(senti_df_2, "Emotion Distribution of MentalManip_maj", "draw_senti_scores_maj.pdf",
                        y_height=1100, font_size=12)

    # draw heat maps of techniques and vulnerabilities
    draw_heatmap(tech_list_1, flag='tech')
    draw_heatmap(vul_list_1, flag='vul')
    draw_heatmap([tech_list_1, vul_list_1], flag='tech_vul')

    draw_heatmap(tech_list_2, flag='tech')
    draw_heatmap(vul_list_2, flag='vul')
    draw_heatmap([tech_list_2, vul_list_2], flag='tech_vul')

    # draw embedding space
    embedding_space([manip_dialogues_1, nonmanip_dialogues_1])
    embedding_space([manip_dialogues_2, nonmanip_dialogues_2])
