#-*- coding: utf-8 -*-
from flask import Flask, render_template, request
import pickle
import numpy as np
import streamlit as st
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
import torch
import json
from transformers import PreTrainedTokenizerFast

from krwordrank.sentence import summarize_with_sentences

app = Flask(__name__)





@app.route('/', methods = ['GET', 'POST'] )
def main():
    
    text_original = ""
    text_summary = ""
    output = ""
    trans_text = ""
    text_keyword = ""
    long_text = []
    
        
    
    if request.method == 'POST':
        # with open("test.json", "rt", encoding='UTF8') as json_file:
        #     json_data = json.load(json_file)
        #     text_original = json_data[5]['text']
        #     text_summary = json_data[5]['text']
        
        text_original = request.form['char1']
        text_summary = request.form['char1']
            
        text_summary = text_summary.replace('\n', "")
        text_summary = text_summary.split()
        length = (len(text_summary) // 500) + 1
        length = (len(text_summary) // length) + 1
        long_text = [text_summary[i:i+length] for i in range(0, len(text_summary), length)]
        
        for i in long_text:
            i = ' '.join(i)
            input_ids = tokenizer.encode(i)
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.unsqueeze(0)
            trans_text = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
            #trans_text = model.generate(input_ids, do_sample=True, max_length = 50, top_k = 50, top_p = 0.95, num_return_sequences=3)
            trans_text = tokenizer.decode(trans_text[0], skip_special_tokens=True)
            output += trans_text
        
        penalty = lambda x:0 if (25 <= len(x) <= 80) else 1
        stopwords = {'영화', '관람객', '너무', '정말', '진짜', '한다.', 'and'}

        keywords, sents = summarize_with_sentences(
           [text_original],
            penalty=penalty,
            stopwords = stopwords,
            diversity=0.5,
            num_keywords=5,
            num_keysents=10,
            verbose=False
        )
        for key in keywords.keys():
            text_keyword += key + ", "
        
        text_keyword = text_keyword[:len(text_keyword)-2]

    return render_template('index.html', original = text_original, summary = output, keyword = text_keyword)


if __name__ == "__main__":
    model = BartForConditionalGeneration.from_pretrained('Seonguk/textSummarization')
    tokenizer = PreTrainedTokenizerFast.from_pretrained('Seonguk/textSummarization')
    app.run(debug=True)