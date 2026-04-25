import numpy as np
import torch
import argparse
import logging
import time
import pdb
import os
import json
import random
from utils import (
    evaluate
)
from tqdm import tqdm
from src.biosyn import (
    QueryDataset, 
    CandidateDataset, 
    DictionaryDataset,
    TextPreprocess, 
    RerankNet, 
    BioSyn
)

LOGGER = logging.getLogger()


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Biosyn train')

    # Required
    parser.add_argument('--model_name_or_path', required=True,
                        help='Directory for pretrained model')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')
    
    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--seed',  type=int, 
                        default=0)
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--draft',  action="store_true")
    parser.add_argument('--topk',  type=int, 
                        default=20)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=16, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=10, type=int)
    parser.add_argument('--initial_sparse_weight',
                        default=0, type=float)
    parser.add_argument('--dense_ratio', type=float,
                        default=0.5)
    parser.add_argument('--save_checkpoint_all', action="store_true")

    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_dictionary(dictionary_path):
    """
    load dictionary
    
    Parameters
    ----------
    dictionary_path : str
        a path of dictionary
    """
    dictionary = DictionaryDataset(
            dictionary_path = dictionary_path
    )
    
    return dictionary.data
    
def load_queries(data_dir, filter_composite, filter_duplicate, filter_cuiless):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    filter_cuiless : bool
        filter samples with cuiless
    """
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate,
        filter_cuiless=filter_cuiless
    )
    
    return dataset.data

def train(args, data_loader, model):
    LOGGER.info("train!")
    
    train_loss = 0
    train_steps = 0
    model.train()
    for i, data in enumerate(data_loader):
        model.optimizer.zero_grad()
        batch_x, batch_y = data
        batch_pred = model(batch_x)  
        loss = model.get_loss(batch_pred, batch_y)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        train_steps += 1

    #train_loss /= (train_steps + 1e-9)
    return train_loss,train_steps
    
def main(args):
    init_logging()
    init_seed(args.seed)
    print(args)
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # load dictionary and queries\
    
    dataset_type_dict={'bc5cdr-chemical':'chemical','bc5cdr-disease':'disease','ncbi-disease':'disease','CRAFT-go':'function','medmention_diagnosis':'diagnosis',
                   'medmention_function':'function','medmention_treatment':'treatment','CAFETERIA':'food',
                  'CONLL2003':'location','Revised_JNLPBA-ct':'celltype','NERO-bp':'bodypart'}
    train_dictionary={}
    for dataset in dataset_type_dict:
        dataset_type=dataset_type_dict[dataset]
        if dataset_type not in train_dictionary:
            train_dictionary[dataset_type]=[]
        train_dictionary[dataset_type].append(load_dictionary(dictionary_path=f'./datasets/{dataset}/train_dictionary.txt'))
    for dataset_type in train_dictionary:
        train_dictionary[dataset_type]=np.concatenate(train_dictionary[dataset_type],axis=0)

    
    train_queries={}
    for dataset in dataset_type_dict:
        dataset_type=dataset_type_dict[dataset]
        if dataset_type not in train_queries:
            train_queries[dataset_type]=[]
        train_queries[dataset_type].append(load_queries(data_dir = f'./datasets/{dataset}/processed_traindev', filter_composite=True, 
                                                        filter_duplicate=True,filter_cuiless=True))
    for dataset_type in train_queries:
        train_queries[dataset_type]=np.concatenate(train_queries[dataset_type],axis=0)

    
    if args.draft:
        train_dictionary = train_dictionary[:100]
        train_queries = train_queries[:10]
        args.output_dir = args.output_dir + "_draft"
    # filter only names
    #names_in_train_dictionary = train_dictionary[:,0]
    #names_in_train_queries = train_queries[:,0]

    names_in_train_dictionary={}
    all_names_in_train_dictionary=[]
    for dataset_type in train_dictionary:
        names_in_train_dictionary[dataset_type]=train_dictionary[dataset_type][:,0]
        all_names_in_train_dictionary.append(train_dictionary[dataset_type][:,0])
    all_names_in_train_dictionary=np.concatenate(all_names_in_train_dictionary,axis=0)
    
        
    names_in_train_queries={}
    for dataset_type in train_queries:
        names_in_train_queries[dataset_type]=train_queries[dataset_type][:,0]
        print(dataset_type,names_in_train_queries[dataset_type].shape)

    # load BERT tokenizer, dense_encoder, sparse_encoder
    biosyn = BioSyn(
        max_length=args.max_length,
        use_cuda=args.use_cuda,
        initial_sparse_weight=args.initial_sparse_weight
    )

    
    #biosyn.init_sparse_encoder(corpus=names_in_train_dictionary)
    biosyn.init_sparse_encoder(corpus=all_names_in_train_dictionary)
    biosyn.load_dense_encoder(
        model_name_or_path=args.model_name_or_path
    )
    all_names_in_train_dictionary=0
    # load rerank model
    model = RerankNet(
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        encoder = biosyn.get_dense_encoder(),
        sparse_weight=biosyn.get_sparse_weight(),
        use_cuda=args.use_cuda
    )
    
    # embed sparse representations for query and dictionary
    # Important! This is one time process because sparse represenation never changes.
    LOGGER.info("Sparse embedding")
    #train_query_sparse_embeds = biosyn.embed_sparse(names=names_in_train_queries)
    
    train_query_sparse_embeds={}
    for dataset_type in names_in_train_queries:
        train_query_sparse_embeds[dataset_type]=biosyn.embed_sparse(names=names_in_train_queries[dataset_type])
    #train_dict_sparse_embeds = biosyn.embed_sparse(names=names_in_train_dictionary)
    
    train_dict_sparse_embeds={}
    for dataset_type in names_in_train_dictionary:
        train_dict_sparse_embeds[dataset_type]=biosyn.embed_sparse(names=names_in_train_dictionary[dataset_type])
    '''
    train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=train_query_sparse_embeds, 
        dict_embeds=train_dict_sparse_embeds
    )
    '''
    train_sparse_score_matrix={}
    for dataset_type in train_query_sparse_embeds:
        train_sparse_score_matrix[dataset_type]=biosyn.get_score_matrix(
                                                        query_embeds=train_query_sparse_embeds[dataset_type], 
                                                        dict_embeds=train_dict_sparse_embeds[dataset_type]
                                                    )
    train_dict_sparse_embeds=0
    train_query_sparse_embeds=0
    print(1)
    time.sleep(5)
    
    '''
    train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=train_sparse_score_matrix, 
        topk=args.topk
    )
    '''
    train_sparse_candidate_idxs={}
    for dataset_type in train_sparse_score_matrix:
        train_sparse_candidate_idxs[dataset_type] = biosyn.retrieve_candidate(
                                                        score_matrix=train_sparse_score_matrix[dataset_type], 
                                                        topk=args.topk
                                                    )
    
    # prepare for data loader of train and dev
    '''
    train_set = CandidateDataset(
        queries = train_queries, 
        dicts = train_dictionary, 
        tokenizer = biosyn.get_dense_tokenizer(), 
        s_score_matrix=train_sparse_score_matrix,
        s_candidate_idxs=train_sparse_candidate_idxs,
        topk = args.topk, 
        d_ratio=args.dense_ratio,
        max_length=args.max_length
    )
    
    '''
    train_set={}
    for dataset_type in train_queries:
        train_set[dataset_type]=CandidateDataset(
            queries = train_queries[dataset_type], 
            dicts = train_dictionary[dataset_type], 
            tokenizer = biosyn.get_dense_tokenizer(), 
            s_score_matrix=train_sparse_score_matrix[dataset_type],
            s_candidate_idxs=train_sparse_candidate_idxs[dataset_type],
            topk = args.topk, 
            d_ratio=args.dense_ratio,
            max_length=args.max_length
        )

    train_sparse_score_matrix=0
    train_dictionary=0
    train_queries=0
    print(2)

    '''
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    '''
    train_loader={}
    for dataset_type in train_set:
        train_loader[dataset_type] = torch.utils.data.DataLoader(
            train_set[dataset_type],
            batch_size=args.train_batch_size,
            shuffle=True,
        )
    print(3)
    start = time.time()
    for epoch in range(1,args.epoch+1):
        # embed dense representations for query and dictionary for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))
        LOGGER.info("train_set dense embedding for iterative candidate retrieval")
        train_loss = 0
        train_steps = 0
        for dataset_type in names_in_train_queries:
            print(dataset_type)
            train_query_dense_embeds = biosyn.embed_dense(names=names_in_train_queries[dataset_type], show_progress=True)
            train_dict_dense_embeds = biosyn.embed_dense(names=names_in_train_dictionary[dataset_type], show_progress=True)
            train_dense_score_matrix = biosyn.get_score_matrix(
                query_embeds=train_query_dense_embeds, 
                dict_embeds=train_dict_dense_embeds
            )
            train_query_dense_embeds=0
            train_dict_dense_embeds=0
            train_dense_candidate_idxs = biosyn.retrieve_candidate(
                score_matrix=train_dense_score_matrix, 
                topk=args.topk
            )
            # replace dense candidates in the train_set
            train_set[dataset_type].set_dense_candidate_idxs(d_candidate_idxs=train_dense_candidate_idxs)
            loss,steps= train(args, data_loader=train_loader[dataset_type], model=model)
            train_loss += loss
            train_steps += steps
        train_loss /= (train_steps + 1e-9)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss,epoch))
        
        # save model every epoch
        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            biosyn.save_model(checkpoint_dir)
        
        # save model last epoch
        if epoch == args.epoch:
            biosyn.save_model(args.output_dir)
            
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    
if __name__ == '__main__':
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    main(args)
