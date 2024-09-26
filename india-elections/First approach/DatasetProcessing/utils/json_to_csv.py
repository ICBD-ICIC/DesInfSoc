# Use this file to convert the india-election-tweets.json dataset from json to csv format
# Still in progress, but as it is know, it works to calculate the metrics

import json
import csv

# open file for reading
tweets_file = open('dataset/india-election-tweets-reduced.json', 'r')
lines = tweets_file.readlines()

# now we will open a file for writing
tweets_csv = open('dataset/india-election-tweets-reduced.csv', 'w')
csv_writer = csv.writer(tweets_csv)

# headers
header = json.loads(lines[0]).keys()
csv_writer.writerow(header)

for line in lines:
    # Writing data of CSV file
    csv_writer.writerow(json.loads(line).values())

tweets_csv.close()
