import json
import csv

FORMATTED_DATASET = 'outputs/india-election-tweets-formatted.csv'

formatted_dataset = open(FORMATTED_DATASET, 'w', encoding="utf8")
csv_writer = csv.writer(formatted_dataset)
csv_writer.writerow(['id',
                     'created_at',
                     'text',
                     'username',
                     'user_id'])

with open('dataset/india-election-tweets.json', 'r', encoding="utf8") as tweets_file:
    for line in tweets_file:
        json_line = json.loads(line)
        csv_writer.writerow([json_line["id_str"],
                             json_line["created_at"],
                             json_line["text"],
                             json_line["user"]["name"],
                             json_line["user"]["id_str"]])

formatted_dataset.close()
