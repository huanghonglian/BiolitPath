import os
import re
import glob
import json
import argparse
from convert import pubtator2dict_list,reformat_tmvar,reformat_gnorm
from abbr_resolver import Abbr_resolver



def process_abbreviations(content,tagged_doc, abbr_dict):
    """
    Process abbreviations in the tagged document based on whether their full forms
    are recognized as entities.
    
    Returns:
        Updated tagged_doc with abbreviations properly handled
    """
    #print(abbr_dict)
    # Step 1: Check if full forms are recognized as entities
    entities=tagged_doc['entities']
    full_form_entity_status = {}
    for abbr, full_form in abbr_dict.items():
        # Check if the full form exists in any entity type
        found_type = ''
        found_distance=1000
        found_prob=0
        found_id=''
        if full_form not in content:
            full_form=full_form.replace('( ','(').replace(' )',')')
            full_form=full_form.replace('[ ','[').replace(' ]',']')
        if full_form not in content:
            continue
        full_form_loc=[]
        for match in re.finditer(r'\b' + re.escape(full_form) + r'\b',content):
            f_start=match.start()
            f_end=match.end()
            full_form_loc.append([f_start,f_end])
        for entity_type, entity_dicts in entities.items():
            for entity_dict in entity_dicts:
                start=entity_dict['start']
                end=entity_dict['end']
                id_=''
                if entity_type not in ['gene','mutation','species','cellline']:
                    prob=entity_dict['prob']
                    '''
                    end=end+1
                    mention=content[start:end]
                    entity_dict['mention']=mention
                    entity_dict['end']+=1
                    print(entity_dict)
                    '''
                else:
                    if entity_type=='species':
                        prob=1
                    else:
                        prob=0.95
                    if 'id' in entity_dict:
                        id_=entity_dict['id']
                for f_start,f_end in full_form_loc:
                    if f_end<=end and f_start>=start:
                        distance=end-f_end+f_start-start
                        if distance<found_distance:
                            found_type=entity_type
                            found_distance=distance
                            found_prob=prob
                            found_id=id_
                        elif distance==found_distance:
                            if prob>found_prob:
                                found_type=entity_type
                                found_distance=distance
                                found_prob=prob
                                found_id=id_
        # Store the status for this abbreviation
        full_form_entity_status[abbr] = [found_type,found_prob,full_form,found_id]
    add_abbr=[]
    for abbr in full_form_entity_status:
        abbr_status=full_form_entity_status[abbr][:]
        if abbr[-1]=='s':
            add_abbr.append([abbr[:-1],abbr_status])
        else:
            add_abbr.append([abbr+'s',abbr_status])
    for item in add_abbr:
        abbr,abbr_status=item
        full_form_entity_status[abbr]=abbr_status[:]
    
    # Step 2: Process each abbreviation based on full form recognition status
    for abbr, found_result in full_form_entity_status.items():
        found_type,found_prob,full_form,found_id=found_result
        pattern = r'\b' + re.escape(abbr) + r'\b'
        if found_type!='':
            # Case 1: Full form is recognized - find all abbreviations in text and tag them
            
            # Find all occurrences of the abbreviation in the text
            # Use word boundaries to avoid matching parts of larger words
            add_entities=[]
            for match in re.finditer(pattern, content):
                add_start, add_end = match.start(), match.end()
                if_add=True
                for entity_dict in entities[found_type]:
                    start=entity_dict['start']
                    end=entity_dict['end']
                    if not (add_end<=start or end<=add_start):
                        if found_type not in ['gene','species','mutation']:
                            entity_dict['prob']=found_prob
                            entity_dict['mention']=entity_dict['mention'].replace(abbr,full_form)
                        if_add=False
                
                if if_add:
                    if found_type not in ['gene','species','mutation']:
                        add_entities.append({'start': add_start,'end': add_end,'prob':found_prob,'mention':full_form})
                    else:
                        add_entities.append({'start': add_start,'end': add_end,'mention':abbr,'id':found_id})
            
            for add_entity in add_entities:
                #print('*',add_entity)
                entities[found_type].append(add_entity) 
            '''
            for entity_type, entity_dicts in entities.items():
                if entity_type==found_type:
                    continue
                del_indexs=[]
                for mention_idx, entity_dict in enumerate(entity_dicts):
                    if re.search(pattern, entity_dict['mention']):
                        del_indexs.append(mention_idx)
                    else:
                        start=entity_dict['start']
                        end=entity_dict['end']
                        for f_start,f_end in full_form_loc:
                            if not (end<=f_start or f_end<=start):
                                del_indexs.append(mention_idx)
                del_indexs = sorted(list(set(del_indexs)), reverse=True)
                
                for del_index in del_indexs:
                    del entity_dicts[del_index]  
            '''

        '''
        else:
            # Case 2: Full form is not recognized - remove any misrecognized abbreviations
            #print(2)
            for entity_type, entity_dicts in entities.items():
                if entity_type in ['gene','species','mutation']:
                    continue
                del_indexs=[]
                for mention_idx, entity_dict in enumerate(entity_dicts):
                    if re.search(pattern, entity_dict['mention']):
                        del_indexs.append(mention_idx)
                del_indexs = sorted(del_indexs, reverse=True)
                
                for del_index in del_indexs:
                    del entity_dicts[del_index]  
        '''
    return tagged_doc
    
def abbr2fullname(tagged_docs):
    ab3p_path='../Ab3P/identify_abbr'
    abbr_resolver = Abbr_resolver(
            ab3p_path=ab3p_path
        )
    abbr_dict={}
    for tagged_doc in tagged_docs:
        content=tagged_doc['title']+' '+tagged_doc['abstract']
        pmid=tagged_doc['pmid']
        abbr_dict = abbr_resolver.resolve(content)
        tagged_doc=process_abbreviations(content,tagged_doc,abbr_dict)
    return tagged_docs


def ner_merge(tagged_docs,tmvar_docs,gnormplus_docs,cellline_docs):
    mutation_types = ['ProteinMutation', 'DNAMutation', 'SNP']

    tagged_doc_index=0
    del_indexs=[]
    for tagged_doc in tagged_docs:
        pmid=tagged_doc['pmid']
        content=tagged_doc['title']+' '+tagged_doc['abstract']
        for entity_type, entity_dict in tagged_doc['entities'].items():
            for mention_idx, entity in enumerate(entity_dict):
                entity['prob']=tagged_doc['prob'][entity_type][mention_idx][1]
                entity['end']+=1
                start=entity['start']
                end=entity['end']
                entity['mention']=content[start:end]
        tagged_doc['entities']['gene']=[]
        tagged_doc['entities']['species']=[]
        tagged_doc['entities']['mutation']=[]
        tagged_doc['entities']['cellline']=[]
        gnorm_found=False
        #gene_entities=[]
        for gnormplus_doc in gnormplus_docs:
            if gnormplus_doc['pmid']==pmid:
                tagged_doc['entities']['gene']=gnormplus_doc['entities']['gene'][:]
                tagged_doc['entities']['species']=gnormplus_doc['entities']['species'][:]
                content_gnorm=gnormplus_doc['title'].rstrip()+' '+gnormplus_doc['abstract']
                gnorm_found=True
                break 
        if gnorm_found and len(content)!=len(content_gnorm):
            '''
            print(len(content),[content])
            print(len(content_gnorm),[content_gnorm])
            print(yes)
            '''
            if tagged_doc_index not in del_indexs:
                del_indexs.append(tagged_doc_index)
            continue
        #tmvar_found=False
        mutation_entities=[]
        for tmvar_doc in tmvar_docs:
            if tmvar_doc['pmid']==pmid:
                for entity in tmvar_doc['entities']['mutation']: 
                    if entity['subtype'] in mutation_types:
                        mutation_entities.append(entity)
                #content_tmvar=tmvar_doc['title']+' '+tmvar_doc['abstract']
                #tmvar_found=True
                break
        tagged_doc['entities']['mutation']=mutation_entities[:]
        
        #cellline_found=False
        for cellline_doc in cellline_docs:
            if cellline_doc['pmid']==pmid:
                tagged_doc['entities']['cellline']=cellline_doc['entities']['cellline'][:]
                #content_cellline=cellline_doc['title']+' '+cellline_doc['abstract']
                #cellline_found=True
                break 
        tagged_doc_index+=1
        del tagged_doc['prob']
    del_indexs = sorted(del_indexs, reverse=True)
    print(f'{len(del_indexs)} pieces of literature were deleted.')
    for del_index in del_indexs:
        del tagged_docs[del_index]
    return tagged_docs
            
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--case',help='Specify case name')
    args = argparser.parse_args()
    ner_path=f'./case/{args.case}/NERoutput/{args.case}*.json'
    ner_files = glob.glob(ner_path)
    
    output_path=f'./case/{args.case}/NERoutput_m'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for ner_file in ner_files:
        with open(ner_file, 'r', encoding='utf-8') as f:
            tagged_docs = json.load(f)
        base_name=os.path.basename(ner_file)[:-5]
        output_file=ner_file.replace('NERoutput','NERoutput_m')
        print('Processing: '+ner_file)
        tmvar_docs=[]
        gnormplus_docs=[]
        cellline_docs=[]
        output_tmvar_norm=f'./case/{args.case}/tmvar3/{base_name}.PubTator.PubTator'
        if os.path.exists(output_tmvar_norm):
            tmvar_docs = pubtator2dict_list(output_tmvar_norm)
            tmvar_docs = reformat_tmvar(tmvar_docs)
        output_gnormplus_norm=f'./case/{args.case}/gnormplus/{base_name}.PubTator'
        if os.path.exists(output_gnormplus_norm):
            gnormplus_docs = pubtator2dict_list(output_gnormplus_norm)
            gnormplus_docs,cellline_docs=reformat_gnorm(gnormplus_docs)
        tagged_docs=ner_merge(tagged_docs,tmvar_docs,gnormplus_docs,cellline_docs)
        #Handling abbreviations
        tagged_docs=abbr2fullname(tagged_docs)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tagged_docs, f)
        
        
        
if __name__ == "__main__":
    main()