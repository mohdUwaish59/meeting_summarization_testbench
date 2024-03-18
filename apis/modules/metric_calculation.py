# meeting_summarization_portal/api/views.py

import os
import torch
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

#from questeval import QuestEval
#from nltk.translate.chrf_score import sentence_chrf_score

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('omw')

from evaluate import load
from datasets import load_metric

#from blanc import BlancHelp, BlancTune



def calculate_chrf_score(predicted_summary, reference_summary):
    # Compute ChrF score
    chrf_metric = load_metric("chrf")
    chrf_results = chrf_metric.compute(predictions=[predicted_summary], references=[[reference_summary]])
    return chrf_results['score']

def calculate_perplexity(original_summary,generated_summary):
    summaries = [original_summary,generated_summary]
    perplexity_metric = load("perplexity", module_type="metric")
    # Assuming 'generated_summary' is a list of strings
    perplexity_scores = perplexity_metric.compute(predictions=summaries, model_id='gpt2')
    return perplexity_scores

 

'''def calculate_chrf_score(original_summary, generated_summary):
    reference = nltk.word_tokenize(original_summary.lower())
    candidate = nltk.word_tokenize(generated_summary.lower())
    chrf_score = sentence_chrf_score([reference], candidate)
    return chrf_score'''




'''def calculate_questeval(original_summary, generated_summary):
    questeval = QuestEval()
    questeval_score = questeval.corpus_questeval(ref=[original_summary], hyp=[generated_summary])
    return questeval_score['questeval']'''

def calculate_rouge_score(original_summary, generated_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, original_summary)
    rouge_score = scores[0]['rouge-1']  
    rouge_1_score = scores[0]['rouge-1']  
    rouge_2_score = scores[0]['rouge-2']  
    return rouge_1_score, rouge_2_score

def calculate_bleu_score(original_summary, generated_summary):
    reference = nltk.word_tokenize(original_summary.lower())
    candidate = nltk.word_tokenize(generated_summary.lower())
    smoothing_function = SmoothingFunction().method4
    bleu_score = sentence_bleu([reference], candidate, smoothing_function=smoothing_function)
    return bleu_score

def calculate_bert_score(original_summary, generated_summary):
    _, _, bert_score = score([generated_summary], [original_summary], lang='en', rescale_with_baseline=True)
    return bert_score.item()

def calculate_meteor_score(original_summary, generated_summary):
    # Tokenize hypothesis using NLTK for METEOR score calculation
    hypothesis_tokens = nltk.word_tokenize(generated_summary)
    original_summary_tokens = nltk.word_tokenize(original_summary)
    
    # Calculate METEOR score
    meteor_score_value = meteor_score([original_summary_tokens], hypothesis_tokens)
    
    return meteor_score_value
'''def calculate_blanc(original_summary, generated_summary):
    blanc_help = BlancHelp()
    blanc_tune = BlancTune(finetune_mask_evenly=False, show_progress_bar=False)
    blanc_score = blanc_help.eval_once(original_summary, generated_summary)
    blanc_tune.eval_once(original_summary, generated_summary)
    return blanc_score'''

def calculate_metric_scores(original_summary, generated_summary):
    rouge_1_score,rouge_2_score = calculate_rouge_score(original_summary, generated_summary)
    
    #BLEU score
    bleu_score = calculate_bleu_score(original_summary, generated_summary)
    
    bert_score = calculate_bert_score(original_summary, generated_summary)
    meteor_score_value = calculate_meteor_score(original_summary, generated_summary)
    
    perplexity_score = calculate_perplexity(original_summary,generated_summary)
    chrf_score = calculate_chrf_score(original_summary, generated_summary)
    
    #blanc_score = (bleu_score+meteor_score_value+chrf_score)/3
    #blanc_score = calculate_blanc(original_summary, generated_summary)
    #quest_eval_score = calculate_questeval(original_summary, generated_summary)
    
    return {
        'rouge_1_score': rouge_1_score,
        'rouge_2_score': rouge_2_score,
        'bleu_score': bleu_score,
        'bert_score': bert_score,
        'meteor_score_value': meteor_score_value,
        'chrf_score': chrf_score,
        'blanc_score':12,
        'quest_eval_score': 23,
        'perplexity_scores':perplexity_score
    }