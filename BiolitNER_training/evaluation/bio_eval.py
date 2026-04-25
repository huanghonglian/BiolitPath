#!/usr/bin/env python3

import argparse
import re
import os
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

def pubtator_to_bio(pubtator_file_path):

    
    def text_to_tokens(text):
        """Split text into a list of tokens (simple whitespace tokenization)."""
        return text.split()
    
    def get_token_positions(text, tokens):
        """Return the start and end character index of each token in the text."""
        positions = []
        current_pos = 0
        for token in tokens:
            # Skip leading whitespace
            while current_pos < len(text) and text[current_pos].isspace():
                current_pos += 1
            
            start = current_pos
            end = start + len(token) - 1
            positions.append((start, end))
            
            # Advance past the current token
            current_pos = end + 1
        
        return positions
    
    def create_bio_labels(tokens, positions, entities):
        """Assign BIO labels to each token."""
        labels = ['O'] * len(tokens)
        
        for entity in entities:
            ent_start = entity['start']
            ent_end = entity['end']
            ent_type = entity['type']
            
            # Tokens whose span falls inside the entity
            for i, (token_start, token_end) in enumerate(positions):
                if token_start >= ent_start and token_end <= ent_end:
                    # Compare with previous token to detect entity start vs. continuation
                    prev_token_end = positions[i-1][1] if i > 0 else -1
                    if prev_token_end < ent_start - 1:  # First token of entity
                        labels[i] = f'B'
                    else:
                        labels[i] = f'I'
        
        return labels
    
    # Read PubTator file
    documents = []
    with open(pubtator_file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Split into documents by blank lines
    doc_blocks = content.split('\n\n')
    
    for block in doc_blocks:
        if not block.strip():
            continue
        
        lines = block.strip().split('\n')
        
        # Parse document block
        doc_info = {}
        entities = []
        
        for line in lines:
            if line.startswith('|t|') or '|t|' in line:
                # Title field
                parts = line.split('|t|')
                doc_info['id'] = parts[0]
                doc_info['title'] = '|t|'.join(parts[1:]) if len(parts) > 1 else ''
            elif line.startswith('|a|') or '|a|' in line:
                # Abstract field
                parts = line.split('|a|')
                doc_info['abstract'] = '|a|'.join(parts[1:]) if len(parts) > 1 else ''
            elif '\t' in line and len(line.split('\t')) >= 4:
                # Entity annotation line
                parts = line.split('\t')
                entity = {
                    'doc_id': parts[0],
                    'start': int(parts[1]),
                    'end': int(parts[2]) - 1,  # Use inclusive end index
                    'text': parts[3],
                    'type': ''
                }
                entities.append(entity)
        
        if doc_info:
            # Full text is title plus abstract
            full_text = doc_info.get('title', '') + ' ' + doc_info.get('abstract', '')
            doc_info['full_text'] = full_text
            doc_info['entities'] = entities
            documents.append(doc_info)
    
    # Build BIO-formatted lines
    bio_lines = []
    i=0
    for doc in documents:
        text = doc['full_text']
        tokens = text_to_tokens(text)
        positions = get_token_positions(text, tokens)
        '''
        if i==42:
            print(tokens)
            print(positions)
        '''
        i+=1
        labels = create_bio_labels(tokens, positions, doc['entities'])
        # BIO: one line per token
        for token, label in zip(tokens, labels):
            bio_lines.append(f"{token}\t{label}")
        # Blank line between documents
        bio_lines.append('')

    
    return bio_lines
def bio_split_by_sent(file_path):
    bio_lines = pubtator_to_bio(file_path)
    # Build BIO from PubTator file and split by document
    tokens, labels = [], []
    current_tokens = []
    current_labels = []
    
    for line in bio_lines:
        if not line:  # Document boundary
            if current_tokens:
                tokens.append(current_tokens)
                labels.append(current_labels)
                current_tokens = []
                current_labels = []
        else:
            parts = line.split('\t')
            if len(parts) >= 2:
                current_tokens.append(parts[0])
                current_labels.append(parts[1])
    
    if current_tokens:
        tokens.append(current_tokens)
        labels.append(current_labels)
    return tokens, labels

def evaluate_pubtator_vs_bio(gold_bio_file, pred_pubtator_file, output_metrics_file=None):

    gold_tokens, gold_labels = bio_split_by_sent(gold_bio_file)
    # Same conversion for predicted PubTator file
    pred_tokens, pred_labels = bio_split_by_sent(pred_pubtator_file)
    # Ensure document counts match
    if len(gold_tokens) != len(pred_tokens):
        print(f"Warning: document count mismatch. Gold: {len(gold_tokens)}, prediction: {len(pred_tokens)}")
            
        #print(gold_tokens[941])
        min_docs = min(len(gold_tokens), len(pred_tokens))
        gold_tokens = gold_tokens[:min_docs]
        gold_labels = gold_labels[:min_docs]
        pred_tokens = pred_tokens[:min_docs]
        pred_labels = pred_labels[:min_docs]
    for i in range(len(gold_tokens)):
        if pred_tokens[i][0]!=gold_tokens[i][0]:
            print(i,gold_tokens[i],pred_tokens[i])
    all_gold_entities = []
    all_pred_entities = []
    all_gold_labels=[]
    all_pred_labels=[]
    for i in range(len(gold_tokens)):
        all_gold_labels.append(gold_labels[i])
        all_pred_labels.append(pred_labels[i])
        all_gold_entities.extend(gold_tokens[i])
        all_pred_entities.extend(pred_tokens[i])
    for i in range(len(all_gold_labels)):
        if len(all_gold_labels[i])!=len(all_pred_labels[i]):
            print(i,len(all_gold_labels[i]),len(len(all_pred_labels[i])))
            break
            
    precision=precision_score(all_gold_labels, all_pred_labels)
    recall= recall_score(all_gold_labels, all_pred_labels)
    f1=f1_score(all_gold_labels, all_pred_labels)
    metrics=[precision,recall,f1]
    
    return metrics

def main():
    models=['biolitpath','bern2','hunflair2']
    entity_types={'ncbi-disease':'disease','BC4CHEMD':'chemical','Revised_JNLPBA-ct':'celltype',
                  'CAFETERIA':'food','CONLL2003':'location','CRAFT-go':'go','Medmention-diagnosis':'diagnosis',
            'Medmention-treatment':'treatment','NERO-bp':'bodypart'}
    eval_results=[]
    print('Evaluating models...')
    for dataset in entity_types:
        et=entity_types[dataset]
        file_paths={}
        gold_file=f'./goldstandard/{dataset}.txt'
        for model in models:
            model_pred=f'./{model}/{dataset}.txt'
            if not os.path.exists(model_pred):
                continue
            result=evaluate_pubtator_vs_bio(gold_file, model_pred)
            result=[round(num, 3) for num in result]
            eval_results.append([dataset,et,model]+result)

    columns=['Dataset','Type','Model']+['Precision','Recall','F1']
    eval_results=pd.DataFrame(eval_results,columns=columns)
    print(eval_results)
    
    
if __name__ == "__main__":
    main()