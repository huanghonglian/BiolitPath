from datetime import datetime
import os
import sys
import time
import glob
import json
import socket
import threading
import argparse


from normalizers.chemical_normalizer import ChemicalNormalizer
from normalizers.species_normalizer import SpeciesNormalizer
from normalizers.cellline_normalizer import CellLineNormalizer
from normalizers.celltype_normalizer import CellTypeNormalizer
from normalizers.neural_normalizer import NeuralNormalizer
from normalizers.dictionary_normalizer import DictNormalizer
#from normalizers.convert import pubtator2dict_list,reformat_tmvar,reformat_gnorm, abbr2fullname


time_format = '[%d/%b/%Y %H:%M:%S.%f]'


class Normalizer:
    def __init__(self, use_neural_normalizer, gene_port=18888, disease_port=18892, no_cuda=False):
        # Normalizer paths
        self.BASE_DIR = 'resources/normalization/'
        self.NORM_INPUT_DIR = {
            'disease': os.path.join(self.BASE_DIR, 'inputs/disease'),
            'gene': os.path.join(self.BASE_DIR, 'inputs/gene'),
        }
        self.NORM_OUTPUT_DIR = {
            'disease': os.path.join(self.BASE_DIR, 'outputs/disease'),
            'gene': os.path.join(self.BASE_DIR, 'outputs/gene'),
        }
        self.NORM_DICT_PATH = {
            'chemical': os.path.join(self.BASE_DIR,
                                'dictionary/dict_ChemicalCompound_202602.txt'),
            'disease': os.path.join(self.BASE_DIR,
                                    'dictionary/dict_disease_202602.txt'),
            'cellline': os.path.join(self.BASE_DIR,
                                    'dictionary/dict_cellline_202602.txt'),
            'celltype': os.path.join(self.BASE_DIR,
                                    'dictionary/dict_CellType_202601.txt'),
            'food': os.path.join(self.BASE_DIR,
                                    'dictionary/dict_food_202512.txt'),
            'go': os.path.join(self.BASE_DIR,
                                    'dictionary/dict_gene_ontology_202512.txt'),
            'treatment': os.path.join(self.BASE_DIR,
                                    'dictionary/dict_treatment_202512.txt'),
            'diagnosis': os.path.join(self.BASE_DIR,
                                    'dictionary/dict_diagnosis_202512.txt'),
            'bodypart': os.path.join(self.BASE_DIR,
                                    'dictionary/dict_uberon_202601.txt'),
            'location': os.path.join(self.BASE_DIR,
                                    'dictionary/dict_geoname_202601.txt'),
        }
        


        # checkpoint on huggingface hub
        self.NEURAL_NORM_MODEL_PATH = 'model/biolitNEN'
        self.NEURAL_NORM_CACHE_PATH = {
            'disease':os.path.join(self.BASE_DIR,
                    'normalizers/neural_norm_caches/dict_disease_202602.txt.pk'),
            'chemical':os.path.join(self.BASE_DIR,
                    'normalizers/neural_norm_caches/dict_ChemicalCompound_202602.txt.pk'),
            'celltype':os.path.join(self.BASE_DIR,
                    'normalizers/neural_norm_caches/dict_CellType_202601.txt.pk'),
            'food':os.path.join(self.BASE_DIR,
                    'normalizers/neural_norm_caches/dict_food_202512.txt.pk'),
            'go':os.path.join(self.BASE_DIR,
                    'normalizers/neural_norm_caches/dict_gene_ontology_202512.txt.pk'),
            'treatment':os.path.join(self.BASE_DIR,
                    'normalizers/neural_norm_caches/dict_treatment_202512.txt.pk'),
            'diagnosis':os.path.join(self.BASE_DIR,
                    'normalizers/neural_norm_caches/dict_diagnosis_202512.txt.pk'),
            'bodypart':os.path.join(self.BASE_DIR,
                    'normalizers/neural_norm_caches/dict_uberon_202601.txt.pk'),
            'location':os.path.join(self.BASE_DIR,
                    'normalizers/neural_norm_caches/dict_geoname_202601.txt.pk'),
        }
        
        self.NORM_MODEL_VERSION = 'biolitNEN v2026'

        self.HOST = '127.0.0.1'

        # normalizer port
        self.GENE_PORT = gene_port
        self.DISEASE_PORT = disease_port

        self.NO_ENTITY_ID = 'undefined'
        self.normalizer={}
        self.normalizer['chemical']= ChemicalNormalizer(self.NORM_DICT_PATH['chemical'])
        for ent_type in ['disease','celltype','food','go','treatment','diagnosis','bodypart','location','cellline','food']:
            self.normalizer[ent_type]=DictNormalizer(self.NORM_DICT_PATH[ent_type])
        # neural normalizer
        self.neural_disease_normalizer = None
        self.neural_chemical_normalizer = None
        self.neural_gene_normalizer = None
        self.use_neural_normalizer = use_neural_normalizer
        
        if self.use_neural_normalizer:
            print("start loading neural normalizer..")
            self.neural_normalizer = NeuralNormalizer(
                model_name_or_path=self.NEURAL_NORM_MODEL_PATH,
                cache_path=self.NEURAL_NORM_CACHE_PATH,
                no_cuda=no_cuda,
            )
            print(f"neural_multi_normalizer is loaded.. model={self.NEURAL_NORM_MODEL_PATH} , dictionary=normalizers/neural_norm_caches")
            '''
            self.neural_chemical_normalizer = NeuralNormalizer(
                model_name_or_path=self.NEURAL_NORM_MODEL_PATH,
                cache_path=self.NEURAL_NORM_CACHE_PATH['chemical'],
                no_cuda=no_cuda,
            )
            print(f"neural_chemical_normalizer is loaded.. model={self.NEURAL_NORM_MODEL_PATH['drug']} , dictionary={self.NEURAL_NORM_CACHE_PATH['drug']}")

            self.neural_gene_normalizer = NeuralNormalizer(
                model_name_or_path=self.NEURAL_NORM_MODEL_PATH['gene'],
                cache_path=self.NEURAL_NORM_CACHE_PATH['gene'],
                no_cuda=no_cuda,
            )
            print(f"neural_gene_normalizer is loaded.. model={self.NEURAL_NORM_MODEL_PATH['gene']} , dictionary={self.NEURAL_NORM_CACHE_PATH['gene']}")
            '''

    def normalize(self, base_name, doc_dict_list):
        start_time = time.time()

        names = dict()
        saved_items = list()
        ent_cnt = 0
        abs_cnt = 0

        for item in doc_dict_list:
            # Get json values
            if 'title' in item:
                content=item['title']
            else:
                content=''
            content =content+' '+item['abstract']
            # pmid = item['pmid']
            entities = item['entities']

            abs_cnt += 1
            # Iterate entities per abstract
            for ent_type, locs in entities.items():
                ent_cnt += len(locs)
                if ent_type in ['gene','species','mutation']:
                    continue
                for loc in locs:
                    if ent_type == 'mutation':
                        name = loc['normalizedName']

                        if ';' in name:
                            name = name.split(';')[0]
                    else:
                        name = loc['mention']

                    if ent_type in names:
                        names[ent_type].append([name, len(saved_items)])
                    else:
                        names[ent_type] = [[name, len(saved_items)]]

            # Work as pointer
            item['norm_model'] = self.NORM_MODEL_VERSION
            saved_items.append(item)
        # For each entity,
        # 1. Write as input files to normalizers
        # 2. Run normalizers
        # 3. Read output files of normalizers
        # 4. Remove files
        # 5. Return oids
        # Threading
        results = list()
        threads = list()
        for ent_type in names.keys():
            t = threading.Thread(target=self.run_normalizers_wrap,
                                 args=(ent_type, base_name, names, saved_items, results))
            t.daemon = True
            t.start()
            threads.append(t)

        # block until all tasks are done
        for t in threads:
            t.join()
        # Save oids
        for ent_type, type_oids in results:
            oid_cnt = 0
            for saved_item in saved_items:
                for loc in saved_item['entities'][ent_type]:
                    # Put oid
                    #names[ent_type][oid_cnt].append(type_oids[oid_cnt])
                    loc['id'] = type_oids[oid_cnt]
                    loc['is_neural_normalized'] = False
                    oid_cnt += 1

        print(datetime.now().strftime(time_format),
              '[{}] Rule-based normalization '
              '{:.3f} sec ({} article(s), {} entity type(s))'
              .format(base_name, time.time() - start_time, abs_cnt,
                      len(names.keys())))

        return saved_items

    # normalize using neural model
    def neural_normalize(self, tagged_doc):
        if 'title' in tagged_doc:
            content=tagged_doc['title']
        else:
            content=''
        content += ' '+tagged_doc['abstract']
        for ent_type in tagged_doc['entities']:
            if ent_type in ['gene','species','mutation','cellline']:
                continue
            entities = tagged_doc['entities'][ent_type]
            entity_names = [e['mention'] for e in entities]
            cuiless_entity_names = []
            for entity, entity_name in zip(entities, entity_names):
                if entity['id'] in [self.NO_ENTITY_ID,'']:
                    cuiless_entity_names.append(entity_name)
            cuiless_entity_names = list(set(cuiless_entity_names))
            if len(cuiless_entity_names) == 0:
                continue
            #print(f"# cui-less in {ent_type}={len(cuiless_entity_names)}")
            norm_entities = self.neural_normalizer.normalize(
                    names=cuiless_entity_names, 
                    ent_type=ent_type
                )
            cuiless_entity2norm_entities = {c:n for c, n in zip(cuiless_entity_names,norm_entities)}
            for entity, entity_name in zip(entities, entity_names):
                if entity_name in cuiless_entity2norm_entities:
                    cui = cuiless_entity2norm_entities[entity_name][0]
                    entity['id'] = cui if cui != -1 else self.NO_ENTITY_ID
                    entity['is_neural_normalized'] = True
                else:
                    entity['is_neural_normalized'] = False
        return tagged_doc

    def run_normalizers_wrap(self, ent_type, base_name, names, saved_items, results):
        results.append((ent_type,
                        self.run_normalizer(ent_type, base_name, names, saved_items)))

    def run_normalizer(self, ent_type, base_name, names, saved_items):
        start_time = time.time()
        name_ptr = names[ent_type]
        oids = list()
        bufsize = 4

        base_thread_name = base_name
        input_filename = base_thread_name + '.concept'
        output_filename = base_thread_name + '.oid'

        print(f'ent_type = {ent_type}')
        
        if ent_type == 'chemical':
            names = [ptr[0] for ptr in name_ptr]
            preds = self.normalizer[ent_type].normalize(names)
            #preds = self.chemical_normalizer.normalize(names)
            for pred in preds:
                oids.append(pred) 
        else:
            names = [ptr[0] for ptr in name_ptr]
            #preds = self.celltype_normalizer.normalize(names)
            preds = self.normalizer[ent_type].normalize(names)
            for pred in preds:
                if pred != self.NO_ENTITY_ID:
                    oids.append(pred)
                else:
                    oids.append(self.NO_ENTITY_ID)

        # 5. Return oids
        assert len(oids) == len(name_ptr), '{} vs {} in {}'.format(
            len(oids), len(name_ptr), ent_type)

        # double checking
        if 0 == len(oids):
            return oids

        cui_less_count = 0
        for oid in oids:
            if self.NO_ENTITY_ID == oid:
                cui_less_count += 1

        print(datetime.now().strftime(time_format),
              '[{}] [{}] {:.3f} sec, undefined: {:.1f}% ({}/{})'.format(
                  base_name, ent_type, time.time() - start_time,
                  cui_less_count * 100. / len(oids),
                  cui_less_count, len(oids)))
        return oids
def resolve_overlap(tagged_docs):
    """
    Resolve overlapping entity annotations by keeping entities with higher probability.
    """
    mutation_types = ['ProteinMutation', 'DNAMutation', 'SNP']
    
    # Flatten all entities into a single list with their types
    for tagged_doc in tagged_docs:
        pmid=tagged_doc['pmid']
        content=tagged_doc['title']+' '+tagged_doc['abstract']
        all_entities = []
        
        for entity_type, entity_dict in tagged_doc['entities'].items():
            for mention_idx, entity in enumerate(entity_dict):
                entity['type']=entity_type
                entity['check_id']=1 if entity['id'] not in ['CUI-less','','undefined'] else 0
                if entity_type in ['mutation','species']:
                    entity['prob']=1
                    if entity['id'] in ['CUI-less','','undefined'] or entity['id'] is None:
                        entity['prob']=0.8
                elif entity_type in ['cellline']:
                    entity['prob']=0.95
                elif entity_type in ['gene']:
                    if entity['id'] in ['CUI-less','','undefined'] or entity['id'] is None:
                        entity['prob']=0.8
                    else:
                        entity['prob']=0.95
                all_entities.append(entity)
        
        # Sort by probability in descending order
        all_entities.sort(key=lambda x: (x['prob'],x['check_id'],-x['start'],x['end']), reverse=True)
        
        
        # Keep track of selected entities
        selected_entities = []
        
        # Iterate through entities and keep non-overlapping ones
        for entity in all_entities:
            '''
            if entity['prob']<0.7:
                continue
            '''
            is_overlapping = False
            
            # Check if current entity overlaps with any selected entity
            for selected in selected_entities:
                if not (entity['end'] <= selected['start'] or entity['start'] >= selected['end']):
                    # Overlapping detected
                    #if entity['prob'] <0.85 or entity['check_id']==0:
                    is_overlapping = True
                    break
            # If no overlap, add to selected entities
            if not is_overlapping:
                selected_entities.append(entity)
            
            
        # Reconstruct the result dictionary
        update_dict={}
        for entity in selected_entities:
            
            
            entity_type=entity['type']
            if entity_type not in update_dict:
                update_dict[entity_type]=[]
            if entity['id'] in ['CUI-less','']:
                entity['id']="undefined"
            if entity['type'] in ['mutation','species','gene','cellline']:
                entity['prob']="null"
            del entity['type']
            del entity['check_id']
            update_dict[entity_type].append(entity)
        tagged_doc['entities']=update_dict
        
        
    return tagged_docs


        
def count_entities(data):
    num_entities = 0
    for d in data:
        if 'entities' not in d:
            continue
        for ent_type, entities in d['entities'].items():
            num_entities += len(entities)

    return num_entities

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--case',help='Specify case name')
    args = argparser.parse_args()
    ner_path=f'./case/{args.case}/NERoutput_m/{args.case}*.json'
    ner_files = glob.glob(ner_path)
    
    normalizer = Normalizer(
            use_neural_normalizer = True
        )
    if not os.path.exists(f'./case/{args.case}/NENoutput'):
        os.makedirs(f'./case/{args.case}/NENoutput')
        
    for ner_file in ner_files:
        with open(ner_file, 'r', encoding='utf-8') as f:
            tagged_docs = json.load(f)
        base_name=os.path.basename(ner_file)[:-5]
        # Rule-based Normalization models
        #for tagged_docs in data:
        num_entities=count_entities(tagged_docs)
        r_norm_start_time = time.time()
        if num_entities > 0:
            tagged_docs = normalizer.normalize(base_name, tagged_docs)
        r_norm_elapse_time = time.time() - r_norm_start_time
        start_time=time.time()
        if normalizer.use_neural_normalizer and num_entities > 0:
            for tagged_doc in tagged_docs:
                tagged_doc = normalizer.neural_normalize(
                    tagged_doc=tagged_doc
                )
        print(datetime.now().strftime(time_format),
          '[{}] neural-based normalization '
          '{:.3f} sec'
          .format(base_name, time.time() - start_time))
        
        '''
        output_tmvar_norm=f'./case/{args.case}/tmvar3/{base_name}.PubTator.PubTator'
        tmvar_docs = pubtator2dict_list(output_tmvar_norm)
        tmvar_docs = reformat_tmvar(tmvar_docs)
        output_gnormplus_norm=f'./case/{args.case}/gnormplus/{base_name}.PubTator'
        gnormplus_docs = pubtator2dict_list(output_gnormplus_norm)
        gnormplus_docs,cellline_docs=reformat_gnorm(gnormplus_docs) 
        num_entities=count_entities(cellline_docs)
        if num_entities > 0:
            cellline_docs = normalizer.normalize(base_name, cellline_docs)
        '''

        tagged_docs=resolve_overlap(tagged_docs)
        output_path=ner_file.replace('NERoutput_m','NENoutput')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tagged_docs, f)
        
        
        
if __name__ == "__main__":
    main()
    