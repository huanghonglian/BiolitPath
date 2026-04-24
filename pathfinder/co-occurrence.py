import os
import sys
import glob
import json
import math
import argparse
import numpy as np
from tqdm import tqdm
import scipy.stats as stats

from nltk.tokenize import sent_tokenize
from ops import json_to_sent


def read_dictionary():
    BASE_DIR='../resources/normalization'

    NORM_DICT_PATH = {
            'chemical': os.path.join(BASE_DIR,'dictionary/dict_ChemicalCompound_202602.txt'),
            'disease': os.path.join(BASE_DIR,'dictionary/dict_disease_202602.txt'),
            'cellline': os.path.join(BASE_DIR,'dictionary/dict_cellline_202602.txt'),
            'celltype': os.path.join(BASE_DIR,'dictionary/dict_CellType_202601.txt'),
            'food': os.path.join(BASE_DIR,'dictionary/dict_food_202512.txt'),
            'go': os.path.join(BASE_DIR,'dictionary/dict_gene_ontology_202512.txt'),
            'treatment': os.path.join(BASE_DIR,'dictionary/dict_treatment_202512.txt'),
            'diagnose': os.path.join(BASE_DIR,'dictionary/dict_diagnosis_202512.txt'),
            'bodypart': os.path.join(BASE_DIR,'dictionary/dict_uberon_202601.txt'),
            'location': os.path.join(BASE_DIR,'dictionary/dict_geoname_202601.txt'),
            'species': os.path.join(BASE_DIR,'dictionary/dict_Species.txt'),
            'gene': '../GNormPlusJava/Dictionary/GeneIDs.txt',
        }
    
    return NORM_DICT_PATH


def assign_entities_to_sentences(json_data):
    mentions_dict={}
    sent_ids=[]
    num=0
    file_sents_pos=[]
    for each_item in json_data:
        #sents=json_to_sent([each_item])
        pmid=each_item['pmid']
        title=each_item['title']
        abstract=each_item['abstract']
        content=title+' '+abstract
        sents = sent_tokenize(content)
        #sents = adjust_sentence_boundaries(content,sents)
        #sents=sents[pmid]['sentence']
        sents=[sent.strip() for sent in sents]
        sents_pos=[]
        sent_num=1
        last_end=0
        for sent in sents:                
            start=content[last_end:].index(sent)+last_end
            end=start+len(sent)
            last_end=end
            if sent[0].islower() and len(sents_pos)!=0:
                sents_pos[-1][1]=end
            else:
                sents_pos.append([start,end,pmid+'_s'+str(sent_num)])
                sent_ids.append(pmid+'_s'+str(sent_num))
                sent_num+=1
            #print([pmid,start,end,sent])
        file_sents_pos+=sents_pos
        for entity_type in each_item['entities']:
            if entity_type not in mentions_dict:
                mentions_dict[entity_type]={}
            for entity in each_item['entities'][entity_type]:
                start=entity['start']
                end=entity['end']
                nor_id=entity['id']
                if nor_id in ['undefined','CUI-less','']:
                    #print(entity_type,[content[start:end+1]],entity)
                    continue
                nor_id=entity_type+':'+nor_id
                if nor_id=='species:9606':
                    continue
                if entity_type not in ['gene','mutation','species','cellline'] and entity['prob']<0.5:
                    continue
                
                sent_id=''
                for pos in sents_pos:
                    if start>=pos[0] and end<=pos[1]:
                        sent_id=pos[2]
                        break
                if sent_id=='':
                    num+=1
                    #print(pmid,entity_type,[content[start:end+1]],entity)
                    continue
                
                if nor_id not in mentions_dict[entity_type]:
                    mentions_dict[entity_type][nor_id]=[]
                if sent_id not in mentions_dict[entity_type][nor_id]:
                    mentions_dict[entity_type][nor_id].append(sent_id)
    sent_ids=list(set(sent_ids))
    #print(num)
    return mentions_dict, sent_ids, file_sents_pos

def assign_entities_to_pmid(json_data):
    mentions_dict={}
    pmids=[]
    num=0
    file_sents_pos=[]
    for each_item in json_data:
        #sents=json_to_sent([each_item])
        pmid=each_item['pmid']
        title=each_item['title']
        abstract=each_item['abstract']
        content=title+' '+abstract
        entity_exist=False
        for entity_type in each_item['entities']:
            if entity_type not in mentions_dict:
                mentions_dict[entity_type]={}
            for entity in each_item['entities'][entity_type]:
                nor_id=entity['id']
                if nor_id in ['undefined','CUI-less','']:
                    continue
                entity_exist=True
                nor_id=entity_type+':'+nor_id
                if nor_id not in mentions_dict[entity_type]:
                    mentions_dict[entity_type][nor_id]=[]
                if pmid not in mentions_dict[entity_type][nor_id]:
                    mentions_dict[entity_type][nor_id].append(pmid)
        if entity_exist:
            pmids.append(pmid)
    return mentions_dict, pmids

    
def biology_relation():
    relation_list=[]
    with open('relation.txt',encoding='utf-8') as fp:
        for line in fp:
            line=line.strip().split('\t')
            if line[2]!='1':
                continue
            relation_list.append('_'.join(sorted(line[:2])))
    return relation_list
def npmi_from_counts(a: int, b: int, c: int, d: int, smooth: float = 0.0):
    """
    Normalized PMI (NPMI)
    Return value range ≈ [-1, 1]
    """
    total = a + b + c + d
    if total <= 0:
        return None

    p_x  = (a + b + smooth) / (total + 4 * smooth)
    p_y  = (a + c + smooth) / (total + 4 * smooth)
    p_xy = (a + smooth)     / (total + 4 * smooth)

    if p_xy == 0 or p_x == 0 or p_y == 0:
        return None 

    pmi = math.log2(p_xy / (p_x * p_y))
    h_xy = -math.log2(p_xy)           

    if h_xy == 0: 
        return 0.0

    npmi = pmi / h_xy

    npmi = max(min(npmi, 1.0), -1.0)

    return npmi

def calculate_cooccurrence_pvalue(case,mentions_dict,sent_num):
    relation_list=biology_relation()
    total=sent_num
    have_calculated_st=[]
    co_occur_data=[]
    have_calculated_cui={}
    for st1 in mentions_dict:
        for st2 in mentions_dict:
            st_merge='_'.join(sorted([st1,st2]))
            if st_merge not in relation_list:
                continue
            if st_merge in have_calculated_st:
                continue
            have_calculated_st.append(st_merge)
            D = dict()
            for cui1 in mentions_dict[st1]:
                sent_list1=set(mentions_dict[st1][cui1])
                #name1=mentions_dict[st1][cui1]['name']
                cui2_2pvalue = dict()
                for cui2 in mentions_dict[st2]:
                    if cui1==cui2:
                        continue
                    cui_merge=tuple(sorted((cui1,cui2)))
                    if cui_merge in have_calculated_cui:
                        continue
                    have_calculated_cui[cui_merge]=''
                    sent_list2=set(mentions_dict[st2][cui2])
                    #name2=mentions_dict[st2][cui2]['name']
                    a=len(sent_list1&sent_list2)
                    b=len(sent_list1-sent_list2)
                    c=len(sent_list2-sent_list1)
                    d=total-a-b-c
                    assert a >= 0 and b >= 0 and c >= 0 and d >= 0, f"{a},{b},{c},{d}四联表值不能为负"
                    assert a + b + c + d == total, f"总和应为{total}，但得到{a+b+c+d}"
                    if a>0:
                        contingency_table = np.array([[a, b], [c, d]])
                        oddsration, pvalue= stats.fisher_exact(contingency_table,alternative='two-sided')
                        if pvalue<1:
                            cui2_2pvalue[cui2]=pvalue
                            D[(cui1,cui2)]=[cui1,st1,cui2,st2,a,b,c,d,pvalue]
                pvalues_list=list(cui2_2pvalue.values())
                if len(pvalues_list)>0:
                    adjusted_pvalue_minmum = min([i for i in pvalues_list if i>0])
                    enrichment_score = [-np.log(p) if p!=0 else -np.log(adjusted_pvalue_minmum)for p in pvalues_list]
                    e_index=0
                    for e, e_pvalue in cui2_2pvalue.items():
                        numerator, denominator = 0, 0
                        for k, k_pvalue in cui2_2pvalue.items():
                            if k_pvalue < 1.0:
                                numerator += 1
                            if k_pvalue < 1.0 and k_pvalue >= e_pvalue:
                                denominator += 1
                        adjusted_pvalue = min(1.0, e_pvalue * float(numerator) / float(denominator))
                        D[(cui1,e)].append(adjusted_pvalue)
                        D[(cui1,e)].append(enrichment_score[e_index])
                        e_index+=1
            if len(D)>0:
                for k in D:
                    row=D[k]
                    co_occur_data.append(row)
    return co_occur_data
            
                        
def merge_dicts(all_mentions):
    all_mentions_dict={}
    if len(all_mentions)==1:
        all_mentions_dict=all_mentions[0]
    else:
        for mentions_dict in all_mentions:
            for entity_type in mentions_dict:
                if entity_type not in all_mentions_dict:
                    all_mentions_dict[entity_type]={}
                for cui,sent_ids in mentions_dict[entity_type].items():
                    if cui not in all_mentions_dict[entity_type]:
                        all_mentions_dict[entity_type][cui]=[]
                    for sent_id in sent_ids:
                        if sent_id not in all_mentions_dict[entity_type][cui]:
                            all_mentions_dict[entity_type][cui].append(sent_id)
    return all_mentions_dict

def normalized_score(co_occur_data):
    edge_weight={}
    node_sum_weight={}
    for row in co_occur_data:
        k=row[0]+'_'+row[2]
        edge_weight[k]=row[-1]
        if row[0] not in node_sum_weight:
            node_sum_weight[row[0]]=0
        node_sum_weight[row[0]]+=row[-1]
        if row[2] not in node_sum_weight:
            node_sum_weight[row[2]]=0
        node_sum_weight[row[2]]+=row[-1]
    for row in co_occur_data:
        k=row[0]+'_'+row[2]
        nor_score=edge_weight[k]/math.sqrt(node_sum_weight[row[0]]+node_sum_weight[row[2]])
        row.append(nor_score)
    return co_occur_data

def should_keep_relation(a, b, c, d, min_support=10):
    """
    Decision tree to determine if a relation should be kept based on co-occurrence statistics.
    """

    if a >= min_support:
        return True
        
    N = a + b + c + d
    # Calculate frequencies
    freq_x = (a + b) / N  # entity X frequency
    freq_y = (a + c) / N  # entity Y frequency
    
    # Calculate conditional probabilities
    p_y_given_x = a / (a + b) if (a + b) > 0 else 0  # P(Y|X)
    p_x_given_y = a / (a + c) if (a + c) > 0 else 0  # P(X|Y)
    max_cooccur_ratio=max(p_y_given_x,p_x_given_y)

        
    # Strong directional relationship
    if max_cooccur_ratio > 0.5:
        return True
    
    # Check 3: At least one high-frequency entity with low co-occurrence
    if freq_x > 0.1 or freq_y > 0.1:
        return False
    
    # Check 4: Both conditional probabilities are too low
    if p_y_given_x < 0.15 and p_x_given_y < 0.15:
        return False
    
    # Default: keep with low confidence
    return True

    


def save_co_occur_data(case,co_occur_data):
    if not os.path.exists(f'../case/{case}/pathfinder'):
        os.makedirs(f'../case/{case}/pathfinder')
    pval_path=f'../case/{case}/pathfinder/co_occur_pvalue.txt'
    edge_path=f'../case/{case}/pathfinder/graph.edgelist'
    node_path=f'../case/{case}/pathfinder/graph.nodelist'
    node_info={}
    with open(pval_path,'w',encoding='utf-8') as fw, open(edge_path,'w',encoding='utf-8') as fw1:
        fw.write('\t'.join(['id1','entity1_type','id2','entity2_type','a','b','c','d','pvalue','adjP','enrichment_score'])+'\n')
        for row in co_occur_data:
            a,b,c,d=row[4:8]
            if row[-2]<0.05:
                keep=should_keep_relation(a,b,c,d)
                if keep:
                    fw1.write('\t'.join([row[0],row[2],str(row[-1])])+'\n')
                    if row[1] not in node_info:
                        node_info[row[1]]={}
                    if row[3] not in node_info:
                        node_info[row[3]]={}
                    if row[0] not in node_info[row[1]]:
                        node_info[row[1]][row[0]]=''
                    if row[2] not in node_info[row[3]]:
                        node_info[row[3]][row[2]]=''
            row=[str(value) for value in row]
            fw.write('\t'.join(row)+'\n')
    norm_dict_path=read_dictionary()
    for entity_type in node_info:
        if entity_type in ['mutation','gene','species']:
            continue
        dict_path=norm_dict_path[entity_type]
        if entity_type=='gene':
            with open(dict_path,encoding='utf-8') as fp:
                for line in fp:
                    name, id_=line.strip().split('\t')
                    id_=id_.split('|')[1]
                    id_=entity_type+':'+id_
                    if id_ in node_info[entity_type]:
                        if node_info[entity_type][id_]=='':
                            node_info[entity_type][id_]=name
        else:
            with open(dict_path,encoding='utf-8') as fp:
                for line in fp:
                    id_,names=line.strip().split('||')
                    id_=entity_type+':'+id_
                    if id_ in node_info[entity_type]:
                        if node_info[entity_type][id_]=='':
                            names=names.split('|')
                            node_info[entity_type][id_]=names[0]
    with open(node_path,'w',encoding='utf-8') as fw:
        for entity_type in node_info:
            for id_, name in node_info[entity_type].items():
                fw.write('\t'.join([id_,name])+'\n')


def save_sent_mentions(case,all_mentions_dict,all_sent_pos):
    if not os.path.exists(f'../case/{case}/pathfinder'):
        os.makedirs(f'../case/{case}/pathfinder')
    sents_path=f'../case/{case}/pathfinder/split_sents_info.txt'
    mentions_path=f'../case/{case}/pathfinder/mentions_in_sents.txt'
    with open(sents_path,'w',encoding='utf-8') as fw:
        fw.write('\t'.join(['start_in_fulltext','end_in_fulltext','sent_id'])+'\n')
        for sent_pos in all_sent_pos:
            sent_pos=[str(i) for i in sent_pos]
            fw.write('\t'.join(sent_pos)+'\n')
    with open(mentions_path,'w',encoding='utf-8') as fw:
        fw.write('\t'.join(['mention_id','sent_ids'])+'\n')
        for entity_type in all_mentions_dict:
            for cui,sent_ids in all_mentions_dict[entity_type].items():
                sent_ids=';'.join(sent_ids)
                fw.write('\t'.join([cui,sent_ids])+'\n')

def save_pmid_mentions(case,all_mentions_dict,):
    if not os.path.exists(f'../case/{case}/pathfinder'):
        os.makedirs(f'../case/{case}/pathfinder')
    mentions_path=f'../case/{case}/pathfinder/mentions_in_pmid.txt'
    with open(mentions_path,'w',encoding='utf-8') as fw:
        fw.write('\t'.join(['mention_id','pmids'])+'\n')
        for entity_type in all_mentions_dict:
            for cui,sent_ids in all_mentions_dict[entity_type].items():
                sent_ids=';'.join(sent_ids)
                fw.write('\t'.join([cui,sent_ids])+'\n')
        

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--case',help='Specify case name',required=True)
    args = argparser.parse_args()
    nen_path=f'../case/{args.case}/NENoutput/{args.case}*.json'
    nen_files = glob.glob(nen_path)
    all_mentions_dict=[]
    all_sent_ids=[]
    all_sent_pos=[]
    all_pmids=[]
    print("Data loading in progress...")
    for nen_file in nen_files:
        print(f'\t{nen_file}')
        with open(nen_file, 'r', encoding='utf-8') as f:
            tagged_docs = json.load(f)
        mentions_dict,sent_ids,sent_pos=assign_entities_to_sentences(tagged_docs)
        #mentions_dict,pmids=assign_entities_to_pmid(tagged_docs)
        all_mentions_dict.append(mentions_dict)
        #all_pmids+=pmids
        all_sent_pos+=sent_pos
        all_sent_ids+=sent_ids
    all_mentions_dict=merge_dicts(all_mentions_dict)
    all_sent_ids=list(set(all_sent_ids))
    #all_pmids=list(set(all_pmids))
    save_sent_mentions(args.case,all_mentions_dict,all_sent_pos)
    #save_pmid_mentions(args.case,all_mentions_dict)

    total_num=len(all_sent_ids)
    print(f"\n[CALCULATING CO-OCCURRENCE] Starting p-value calculation...")
    co_occur_data=calculate_cooccurrence_pvalue(args.case,all_mentions_dict,total_num)
    
    print(f"✓ Calculated co-occurrence for {len(co_occur_data)} entity pairs")
    print(f"\n[SAVING] Writing co-occurrence data to output file...")
    print(f"[SAVE PATH] Results will be saved to: ../case/{args.case}/pathfinder")

    save_co_occur_data(args.case,co_occur_data)
    print(f"✓ Co-occurrence analysis completed successfully for case: '{args.case}'")
    
    
    
if __name__ == "__main__":
    main()

