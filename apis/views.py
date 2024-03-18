# meeting_summarization_portal/api/views.py

import os
import torch
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from .serializers import ModelLinkSerializer
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
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('omw')


import apis.modules.metric as metric  
import apis.modules.metric_calculation as metric_calc 



def load_datasets():
    datasets = []
    artifacts_folder = 'C://Users//Mohd Uwaish//Desktop//MS//SEM2//SEM_2--SDS//Project//MeetingSum_Testbench//artifacts'  
    for filename in os.listdir(artifacts_folder):
        if filename.endswith('.csv'):  
            dataset_path = os.path.join(artifacts_folder, filename)
            # Load dataset into memory (You may need to use pandas or another library)
            # dataset = load_dataset(dataset_path)
            dataset = pd.read_csv(dataset_path)
            datasets.append(dataset)
    return datasets

    



    
def index(request):
    return render(request, 'index.html')

# meeting_summarization_portal/api/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import base64
import io
import matplotlib.pyplot as plt
from .serializers import ModelLinkSerializer

class SummarizationAPIView(APIView):
    def post(self, request, format=None):
        # Deserialize user input
        serializer = ModelLinkSerializer(data=request.data)
        if serializer.is_valid():
            model_link = serializer.validated_data.get('model_link')

            tokenizer = AutoTokenizer.from_pretrained(model_link)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_link)
            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

            dataset_path = 'C://Users//Mohd Uwaish//Desktop//MS//SEM2//SEM_2--SDS//Project//MeetingSum_Testbench//artifacts//augmented_dataset_char_delete.csv'
            dataset = pd.read_csv(dataset_path)

            first_row = dataset.iloc[0]

            source_text = first_row['source']
            original_summary = first_row['summary']

            max_length = tokenizer.model_max_length
            if len(tokenizer.encode(source_text)) > max_length:
                source_text = source_text[:max_length]

            generated_summary = summarizer(source_text, max_length=5000, do_sample=False)[0]['summary_text']

            dataset.at[0, 'generated_summary'] = generated_summary
            metric_scores = metric_calc.calculate_metric_scores(original_summary, generated_summary)
            
            unique_words = metric.calculate_unique_words(first_row['generated_summary'])
            
            total_words = metric.calculate_total_words(first_row['generated_summary'])
            
            ttr = metric.calculate_ttr(first_row['generated_summary'])
            compression_ratio= metric.calculate_compression_ratio(first_row['generated_summary'],first_row['summary'])
            top_unigrams =  metric.calculate_top_n_ngrams(first_row['generated_summary'], 1)
            top_bigrams =  metric.calculate_top_n_ngrams(first_row['generated_summary'], 2)
            top_trigrams =  metric.calculate_top_n_ngrams(first_row['generated_summary'], 3)

            unigrams_data = {' '.join(ngram[0]): ngram[1] for ngram in top_unigrams}
            bigrams_data = {' '.join(ngram[0]): ngram[1] for ngram in top_bigrams}
            trigrams_data = {' '.join(ngram[0]): ngram[1] for ngram in top_trigrams}

            
            
            histogram_data = {
            'compression_ratio': compression_ratio,
            'unigrams': unigrams_data,
            'bigrams': bigrams_data,
            'trigrams': trigrams_data,
            'total_words':total_words,
            'unique_words': unique_words,
            'type_token_ratio': ttr
        }

            # Prepare response data
            response_data = {
                'original_summary': original_summary,
                'generated_summary': generated_summary,
                'metric_scores':metric_scores,
                'histogram_data': histogram_data
            }
            print(response_data)
            
            return Response(response_data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

'''class SummarizationAPIView(APIView):
    def post(self, request, format=None):
        # Deserialize user input
        serializer = ModelLinkSerializer(data=request.data)
        if serializer.is_valid():
            model_link = serializer.validated_data.get('model_link')

            tokenizer = AutoTokenizer.from_pretrained(model_link)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_link)
            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

            dataset_path = 'C://Users//Mohd Uwaish//Desktop//MS//SEM2//SEM_2--SDS//Project//MeetingSum_Testbench//artifacts//augmented_dataset_char_delete.csv'  # Specify the path to your dataset CSV file
            dataset = pd.read_csv(dataset_path)

            first_row = dataset.iloc[0]

            source_text = first_row['source']
            original_summary = first_row['summary']

            max_length = tokenizer.model_max_length
            if len(tokenizer.encode(source_text)) > max_length:
                source_text = source_text[:max_length]

            generated_summary = summarizer(source_text, max_length=5000, do_sample=False)[0]['summary_text']

            dataset.at[0, 'generated_summary'] = generated_summary
            metric_scores = calculate_metric_scores(original_summary, generated_summary)

           
            dataset.to_csv(dataset_path, index=False)

            
            response_data = {
                'source_length': len(source_text),
                'model_max_tokens_accept': max_length,
                'original_summary': original_summary,
                'generated_summary': generated_summary,
                'metric_scores': metric_scores,
                'source_chunk': source_text  
            }
            #return render(request, 'index.html', {'data': response_data})
            return Response(response_data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)'''



         
'''class SummarizationAPIView(APIView):
    def post(self, request, format=None):
        # Deserialize user input
        serializer = ModelLinkSerializer(data=request.data)
        if serializer.is_valid():
            model_link = serializer.validated_data.get('model_link')

            tokenizer = AutoTokenizer.from_pretrained(model_link)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_link)
            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

            dataset_path = 'C://Users//Mohd Uwaish//Desktop//MS//SEM2//SEM_2--SDS//Project//MeetingSum_Testbench//artifacts//augmented_dataset_char_delete.csv'  # Specify the path to your dataset CSV file
            dataset = pd.read_csv(dataset_path)

            first_row = dataset.iloc[0]

            source_text = first_row['source']
            original_summary = first_row['summary']

            max_length = tokenizer.model_max_length
            encoded_source_text = tokenizer.encode(source_text)
            chunk_size = max_length - 100  # Leave some room for the summary tokens

            # Split the encoded text into chunks of the specified size
            chunks = [encoded_source_text[i:i+chunk_size] for i in range(0, len(encoded_source_text), chunk_size)]

            summaries = []
            for chunk in chunks:
                chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)

                generated_summary = summarizer(chunk_text, max_length=max_length, do_sample=False)[0]['summary_text']
                summaries.append(generated_summary)

            # Combine the summaries from all chunks
            combined_summary = ' '.join(summaries)

            dataset.at[0, 'generated_summary'] = combined_summary
            metric_scores = calculate_metric_scores(original_summary, combined_summary)

            dataset.to_csv(dataset_path, index=False)

            response_data = {
                'source_length': len(source_text),
                'model_max_tokens_accept': max_length,
                'original_summary': original_summary,
                'generated_summary': combined_summary,
                'metric_scores': metric_scores,
                'source_chunk': source_text  
            }
            return Response(response_data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)'''


'''import base64
import io

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
import base64
import io
import textacy

class VisualizationAPIView(APIView):
    def post(self, request, format=None):
        # Deserialize user input (assuming the dataset is provided in the request)
        dataset_path = request.data.get('dataset_path')

        # Load dataset
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)

        # Access the first row of the dataset
        first_row = df.iloc[0]

        # Tokenize text and remove stopwords
        nltk.download('punkt')
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

        def preprocess_text(text):
            tokens = word_tokenize(text.lower())
            return [word for word in tokens if word.isalnum() and word not in stop_words]

        # Calculate compression ratio
        compression_ratio = len(first_row['generated_summary']) / len(first_row['summary'])

        # Calculate unique words in summary
        unique_words = len(set(preprocess_text(first_row['generated_summary'])))

        # Calculate word length and number of words in summary
        word_length = [len(word) for word in preprocess_text(first_row['generated_summary'])]
        num_words = len(preprocess_text(first_row['generated_summary']))

        # Calculate top n unigrams, bigrams, and trigrams
        n = 5
        unigrams = FreqDist(nltk.ngrams(preprocess_text(first_row['generated_summary']), 1)).most_common(n)
        bigrams = FreqDist(nltk.ngrams(preprocess_text(first_row['generated_summary']), 2)).most_common(n)
        trigrams = FreqDist(nltk.ngrams(preprocess_text(first_row['generated_summary']), 3)).most_common(n)

        # Visualize histograms
        plt.figure(figsize=(12, 6))

        # Histogram of the number of words
        plt.subplot(2, 2, 1)
        plt.hist(num_words, bins=10, color='skyblue', edgecolor='black')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.title('Histogram of Number of Words in Summary')

        # Histogram of word lengths
        plt.subplot(2, 2, 2)
        plt.hist(word_length, bins=10, color='salmon', edgecolor='black')
        plt.xlabel('Word Length')
        plt.ylabel('Frequency')
        plt.title('Histogram of Word Lengths in Summary')

        # Histogram of compression ratio
        plt.subplot(2, 2, 3)
        plt.hist(compression_ratio, bins=10, color='lightgreen', edgecolor='black')
        plt.xlabel('Compression Ratio')
        plt.ylabel('Frequency')
        plt.title('Histogram of Compression Ratios')

        # Visualize top n unigrams, bigrams, and trigrams
        plt.subplot(2, 2, 4)
        ngram_labels = [f'{n}-gram' for n in range(1, n+1)]
        ngram_values = [unigrams, bigrams, trigrams]
        for i, (ngram, values) in enumerate(zip(ngram_labels, ngram_values), start=1):
            plt.bar([f'{gram[0]}' for gram in values], [gram[1] for gram in values], alpha=0.7, label=ngram)
        plt.xlabel('Top n n-grams')
        plt.ylabel('Frequency')
        plt.title('Top n Unigrams, Bigrams, and Trigrams')
        plt.legend()

        # Save the visualization as bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualization_data = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        # Calculate Type-Token Ratio
       # ttr_value = textacy.text_stats.ttr(preprocess_text(first_row['generated_summary']))

        # Calculate Coleman Liau Index
        #coleman_liau_value = textacy.text_stats.coleman_liau_index(first_row['generated_summary'])

        # Prepare response data
        response_data = {
            'compression_ratio': compression_ratio,
            'unique_words': unique_words,
            'visualization': visualization_data,
            #'ttr_value': ttr_value,
            #'coleman_liau_value': coleman_liau_value,
        }
        return Response(response_data, status=status.HTTP_200_OK)'''
        
        
        


