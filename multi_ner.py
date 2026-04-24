import os
import sys
import json
import argparse

from datetime import datetime
from multi_ner.main import MTNER
from multi_ner.ops import filter_entities, pubtator2dict_list

def count_entities(data):
    num_entities = 0
    for d in data:
        if 'entities' not in d:
            continue
        for ent_type, entities in d['entities'].items():
            num_entities += len(entities)

    return num_entities

def mtner_recognize(model, dict_path, base_name, args):
    input_mt_ner = os.path.join(args.mtner_home, args.case,'pubtator',
                                f'{dict_path}.PubTator')
    output_mt_ner = os.path.join(args.mtner_home,args.case, 'NERoutput',
                                f'{dict_path}.json')
    dict_list = pubtator2dict_list(input_mt_ner)
    if type(dict_list)!=list:
        sys.exit(1)
    res = model.recognize(
        input_dl=dict_list,
        base_name=base_name
    )
    if res is None:
        return None, 0
    #num_entities = count_entities(res)
    #res[0]['num_entities'] = num_entities
    # Write output str to a .PubTator format file
    with open(output_mt_ner, 'w', encoding='utf-8') as f:
        json.dump(res, f)
    

def run_server(model, args):
    output_path=os.path.join(args.mtner_home,args.case, 'NERoutput')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    print(f"NER output saved to: {output_path}")
    for file in os.listdir(os.path.join(args.mtner_home, args.case,'pubtator')):
        if 'PubTator' not in file:
            continue
        '''
        if os.path.exists(os.path.join(args.mtner_home, args.case,'NERoutput',file.replace('PubTator','json'))):
            continue
        '''
        print(f"Processing file: {file}...")
        dict_path=os.path.splitext(file)[0]
        base_name=os.path.splitext(file)[0]
        mtner_recognize(model, dict_path, base_name, args)

    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, help='random seed for initialization',
                            default=1)
    argparser.add_argument('-c','--case', help='Specify the case name')
    argparser.add_argument('--model_name_or_path', default='./model/biolitNER')
    argparser.add_argument('--max_seq_length', type=int, help='The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.',
                            default=128)
    argparser.add_argument('--mtner_home',
                           help='biomedical language model home',
                          default='./case')         
    argparser.add_argument('--time_format',
                            help='time format', default='[%d/%b/%Y %H:%M:%S.%f]')    
    argparser.add_argument('--no_cuda', action="store_true", help="Avoid using CUDA when available")
    args = argparser.parse_args()

    mt_ner = MTNER(args)
    run_server(mt_ner, args)
    

    