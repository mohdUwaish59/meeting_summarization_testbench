import pandas as pd

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from rouge import Rouge

from bert_score import score

import nltk
from nltk.translate.meteor_score import meteor_score

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
import nltk
from nltk.translate.meteor_score import meteor_score
from django.shortcuts import render
from nltk.tokenize import word_tokenize
import string
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('omw')

# Function to calculate number of unique words in summary
def calculate_unique_words(summary):
    tokens = word_tokenize(summary.lower())
    return len(set(tokens))

def calculate_total_words(summary):
    tokens = word_tokenize(summary.lower())
    return len((tokens))

# Function to calculate top n n-grams with stopwords removed
def calculate_top_n_ngrams(text, n):
    tokens = [token.lower() for token in word_tokenize(text)]
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwords]
    ngrams = FreqDist(nltk.ngrams(tokens, n)).most_common(n)
    return ngrams

# Function to calculate Type-Token Ratio (TTR)
def calculate_ttr(text):
    tokens = word_tokenize(text.lower())
    return len(set(tokens)) / len(tokens)

# Function to calculate top n n-grams with stopwords and punctuation removed
def calculate_top_n_ngrams(text, n):
    tokens = [token.lower() for token in word_tokenize(text)]
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwords and token not in string.punctuation]
    ngrams = FreqDist(nltk.ngrams(tokens, n)).most_common()
    return ngrams

def calculate_compression_ratio(original_summary, generated_summary):
    # Calculate the length of original and generated summaries
    original_length = len(original_summary)
    generated_length = len(generated_summary)
    
    # Compute the compression ratio
    compression_ratio = generated_length / original_length if original_length != 0 else 0
    
    return compression_ratio



# Select only the first row for testing purposes
#first_row = dataset.iloc[0]

# Calculate metrics for the first row (testing purposes)
'''unique_words = calculate_unique_words(first_row['generated_summary'])
ttr = calculate_ttr(first_row['generated_summary'])

# Convert the labels to strings for unigrams, bigrams, and trigrams
# Calculate top unigrams, bigrams, and trigrams
top_unigrams = calculate_top_n_ngrams(first_row['generated_summary'], 1)
top_bigrams = calculate_top_n_ngrams(first_row['generated_summary'], 2)
top_trigrams = calculate_top_n_ngrams(first_row['generated_summary'], 3)

# Convert the format of top n-grams data for word cloud generation
unigrams_data = {' '.join(ngram[0]): ngram[1] for ngram in top_unigrams}
bigrams_data = {' '.join(ngram[0]): ngram[1] for ngram in top_bigrams}
trigrams_data = {' '.join(ngram[0]): ngram[1] for ngram in top_trigrams}'''

# Generate word clouds for unigrams, bigrams, and trigrams
###unigrams_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(unigrams_data)
#bigrams_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bigrams_data)
#trigrams_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(trigrams_data)

