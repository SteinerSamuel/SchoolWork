V = FeatureUnion([("count", CountVectorizer(preprocessor=preprocess_text, stop_words=frozenset(nltk.corpus.stopwords.words))), ("POS_Count", CountVectorizer(preprocessor=preprocess_text ,tokenizer=pos_tokenizer)), ("stem_count", CountVectorizer(tokenizer=stem_tokenizer, preprocessor=preprocess_text))])