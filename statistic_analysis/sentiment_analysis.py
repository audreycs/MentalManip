from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
import csv


def get_max_sentiment(result):
    keys_to_extract = ['neg', 'neu', 'pos']
    extracted_items = {k: result[k] for k in keys_to_extract if k in result}
    sorted_items = sorted(extracted_items.items(), key=lambda item: item[1], reverse=True)
    return sorted_items[0][0]


def get_dialogue(file):
    with open(file, 'r') as f:
        dialogue_list = []
        for line in f.readlines():
            parts = line.strip().split('\t')
            sentence_list = parts[3:]
            dialogue = ''
            for s in sentence_list:
                if ':' not in s:
                    sentence = s
                    dialogue += sentence + ' '
                    continue
                character = s.split(':')[0][:-3]
                content = s.split(':')[1]
                sentence = ': '.join([character, content])
                dialogue += sentence + '\n'
            dialogue_list.append(dialogue)
        return dialogue_list


if __name__ == '__main__':
    dialogue_list = get_dialogue('data/ConvoKitMovie/convokit_movie_conversation.txt')
    random.shuffle(dialogue_list)
    classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        top_k=None
    )
    size = 500
    count = 0
    analyzer = SentimentIntensityAnalyzer()
    neutral_list = []
    for d in dialogue_list:
        if count >= size:
            break
        sentiment_scores = analyzer.polarity_scores(d)
        max_sentiment = get_max_sentiment(sentiment_scores)
        if (max_sentiment == 'pos' and sentiment_scores['compound'] > 0.0 and
                sentiment_scores['neu'] < 0.4 and sentiment_scores['neg'] < 0.1):
            count += 1
            print("*" * 30 + f'{count}/{size}' + "*" * 30)
            print(d)
            print(sentiment_scores)
            neutral_list.append(d)

    with open('data/labeled/sent_positive.csv', 'w', newline='', encoding='utf-8') as csvfile:
        columns = ["Dialogue", "Manipulative", "Technique", "Vulnerability"]
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)
        for d in neutral_list:
            d = d.strip()
            writer.writerow([d, '0', '', ''])
