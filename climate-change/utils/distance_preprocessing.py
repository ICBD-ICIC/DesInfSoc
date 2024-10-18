from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

# tweets = union de tweets de influencers y de replies

for tweet in tweets:
    tokens = tokenizer.tokenize(tweet['text'])

    tokens_no_stopwords = [word for word in tokens if word not in stop_words]

    # todo: remove punctuation
    # todo: check code
    pos_tags = pos_tag(tweet['text'])
    lemmatized_words = []
    for word, tag in pos_tags:
        # Map the POS tag to WordNet POS tag
        pos = wordnet_map.get(tag[0].upper(), wordnet.NOUN)
        # Lemmatize the word with the appropriate POS tag
        lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
        # Add the lemmatized word to the list
        lemmatized_words.append(lemmatized_word)

    return lemmatized_words
