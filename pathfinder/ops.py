import re
import copy
import json
import time

import numpy as np
#import xml.etree.ElementTree as ElTree

from datetime import datetime, timezone
from operator import itemgetter


tokenize_regex = re.compile(r'([0-9a-zA-Z]+|[^0-9a-zA-Z])')

def json_to_sent(data):
    '''data: list of json file [{pmid,abstract,title}, ...] '''
    out = dict()
    for paper in data:
        sentences = list()
        if len(CoNLL_tokenizer(paper['title'])) < 50:
            title = [paper['title']]
        else:
            title = sentence_split(paper['title'])
        if len(title) != 1 or len(title[0].strip()) > 0:
            sentences.extend(title)

        if len(paper['abstract']) > 0:
            abst = sentence_split(paper['abstract'])
            if len(abst) != 1 or len(abst[0].strip()) > 0:
                abst[0]=' '+abst[0]
                sentences.extend(abst)
            
        out[paper['pmid']] = dict()
        out[paper['pmid']]['sentence'] = sentences
        
    return out



def CoNLL_tokenizer(text):
    rawTok = [t for t in tokenize_regex.split(text) if t]
    assert ''.join(rawTok) == text
    tok = [t for t in rawTok if t != ' ']
    return tok

def sentence_split(text):
    sentences = list()
    sent = ''
    piv = 0
    for idx, char in enumerate(text):
        if char in "?!":
            if idx > len(text) - 3:
                sent = text[piv:]
                piv = -1
            else:
                sent = text[piv:idx + 1]
                piv = idx + 1

        elif char == '.':
            if idx > len(text) - 3:
                sent = text[piv:]
                piv = -1
            elif (text[idx + 1] == ' ') and (
                    text[idx + 2] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ-"' + "'"):
                sent = text[piv:idx + 1]
                piv = idx + 1

        if sent != '':
            sentences.append(sent)
            sent = ''

            if piv == -1:
                break

    if piv != -1:
        sent = text[piv:]
        sentences.append(sent)
        sent = ''
    '''
    final_sentences=[]
    for sent_idx,sent in enumerate(sentences):
        if len(sent)
    '''
    return sentences

