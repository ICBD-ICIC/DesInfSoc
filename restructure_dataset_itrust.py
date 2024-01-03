import json
import csv

FORMATTED_DATASET = 'outputs/india-election-tweets-formatted.csv'
FAILED_ROWS = 'outputs/india-election-tweets-failed.json'

formatted_dataset = open(FORMATTED_DATASET, 'w', encoding="utf8")
csv_writer = csv.writer(formatted_dataset)
csv_writer.writerow(['id',
                     'created_at',
                     'text',
                     'username',
                     'user_id'])
failed_rows = open(FAILED_ROWS, 'w', encoding="utf8")

with open('dataset/india-election-tweets.json', 'r', encoding="utf8") as tweets_file:
    for line in tweets_file:
        try:
            json_line = json.loads(line)
            csv_writer.writerow([json_line["id_str"],
                                 json_line["created_at"],
                                 json_line["text"],
                                 json_line["user"]["screen_name"],
                                 json_line["user"]["id_str"]])
        except Exception as e:
            print(e)
            print(line)
            json.dump(line, failed_rows)

formatted_dataset.close()
failed_rows.close()
