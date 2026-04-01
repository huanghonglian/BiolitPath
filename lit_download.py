import os
import sys
import csv
import math
import json
import time
import getopt
import requests
import argparse
import pandas as pd
from datetime import date
from datetime import datetime
from Bio import Medline, Entrez
from multi_ner.ops import CoNLL_tokenizer
Entrez.email = ""


def term_to_pmid(term_file: str, save_path=None) -> list:
    """输入查询词，访问PubMed数据库，获得PMID列表."""
    # 构建查询语句
    with open(term_file, 'r') as f:
        term = f.readline().strip('\n')
    print(f'[search term]: {term}')
    handle0 = Entrez.esearch(db='pubmed', retmax=100000,term=term)
    results = Entrez.read(handle0)
    handle0.close()
    count = int(results['Count'])
    if count<=10000:
        all_pmids=results['IdList']
    else:
        year_ranges=[]
        current_year = date.today().year
        if count<=50000:
            year_ranges.append((1970,1999))
            for year in range(2000,current_year,2):
                year_ranges.append((year,year+2))
        else:
            for year in range(1969,1995,5):
                year_ranges.append((year+1,year+5))
            for year in range(2000,current_year+1):
                year_ranges.append((year,year))
        all_pmids=[]
        for start_year, end_year in year_ranges:
            yearly_term = f'{term} AND ({start_year}[PDAT] : {end_year}[PDAT])'
            try:
                handle = Entrez.esearch(
                    db='pubmed',
                    term=yearly_term,
                    retmax=100000,
                    usehistory='y'
                )
                results = Entrez.read(handle)
                handle.close()
                
                count = int(results['Count'])
                
                if count > 0:
                    if count <= 10000 or end_year<2000:
                        all_pmids.extend(results['IdList'])
                    else:
                        quarterly_pmids = get_pmids_by_quarter(yearly_term, start_year, end_year)
                        all_pmids.extend(quarterly_pmids)
                
                time.sleep(0.34)  # 遵守频率限制
                
            except Exception as e:
                print(f"Error while searching {start_year}-{end_year}: {e}")
                continue
    
    all_pmids=list(set(all_pmids))
    # save results
    if save_path:
        with open(save_path, 'w') as f:
            #f.write('PMID\n')
            f.write('\n'.join(all_pmids) + '\n')
    print(f'[pmid]: {len(all_pmids)}, [save path]: {save_path}')
    return all_pmids

def get_pmids_by_quarter(term, start_year, end_year):
    pmids = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            if quarter == 1:
                date_range = f'{year}/01/01:{year}/03/31'
            elif quarter == 2:
                date_range = f'{year}/04/01:{year}/06/30'
            elif quarter == 3:
                date_range = f'{year}/07/01:{year}/09/30'
            else:
                date_range = f'{year}/10/01:{year}/12/31'
            
            quarterly_term = f'{term} AND ({date_range}[PDAT])'
            
            try:
                handle = Entrez.esearch(
                    db='pubmed',
                    term=quarterly_term,
                    retmax=100000,
                    usehistory='y'
                )
                results = Entrez.read(handle)
                handle.close()
                count = int(results['Count'])
                if count > 0:
                    pmids.extend(results['IdList'])
                time.sleep(0.34)
                
            except Exception as e:
                print(f"Error while searching {date_range}: {e}")
                continue
    return pmids

def parse_PMCIDsFile(pmcids_file = "../knol/PubMed/PMC-ids.csv"):
    """
    The PMC-ids.csv.gz file, available through the FTP service,
    maps an article’s standard IDs to each other and to other article metadata elements.
    PMC-ids.csv.gz is a comma-delimited file with the following fields:
    Journal Title, ISSN, ..., PMCID, PubMed ID (if available), ..., Release Date (Mmm DD YYYY or live)
    link: https://www.ncbi.nlm.nih.gov/pmc/pmctopmid/
    """
    pmcid2pmid = dict()
    pmid2pmcid = dict()
    print(f'loading {pmcids_file}, please waiting...')
    with open(pmcids_file, 'r') as inf:
        rows = list(csv.reader(inf))
        for row in rows[1:]:
            pmcid2pmid[row[8]] = row[9]
            if row[9]:
                pmid2pmcid[row[9]] = row[8]
    print(f'{len(pmcid2pmid)} pmcid mapped to {len(pmid2pmcid)} pmid in file PMC-ids.csv.gz')
    return pmcid2pmid, pmid2pmcid


def pmid_to_pmcid(pmids: list, pmid2pmcid: dict, save_path=None):
    """输入PMID，获得相应的PMCID"""
    pmcids = [pmid2pmcid.get(p, '') for p in pmids]
    pmcids_count = len(pmcids) - pmcids.count('')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('PMID\tPMCID\n')
            for pmid, pmcid in zip(pmids, pmcids):
                f.write('{}\t{}\n'.format(pmid, pmcid))
        print(f'[pmcid]: {pmcids_count}, [save path]: {save_path}')
    return pmcids


def download_medline(case, pmids:list=None, save_path=None):
    t1 = time.time()
    # 如果存储路径不存在，则创建
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print("[case]: {}, [pmid]: {}, [save path]:{}".format(case, len(pmids), save_path))
    # 分批次下载，每次下载10000条
    count = len(pmids)
    batch_size = 500
    iterations = [[i * batch_size, min((i + 1) * batch_size, count)] for i in range((count-1) // batch_size + 1)]
    # 开始分批次下载
    for (start, end) in iterations:
        cur_save_path = os.path.join(save_path, "{}{}.txt".format(case,math.ceil(end/500)))
        pubtator_save_path=cur_save_path.replace('medline','pubtator').replace('.txt','.PubTator')
        if os.path.exists(cur_save_path) or os.path.exists(pubtator_save_path):
            print('\t[Downloaded]:{}-{}'.format(start, end))
            continue
        # 获取数据，Medline格式
        handle = Entrez.efetch(db='pubmed', id=pmids[start:end], rettype='medline', retmode='text')
        try:
            records=handle.read()
            handle.close()
            with open(cur_save_path,'w',encoding='utf-8') as f:
                f.write(records)
            print('\t[Downloading]: {}-{}'.format(start, end))
            # 休眠1秒，避免持续访问导致连接中断。
            time.sleep(1)
        except:
            print('not download:',start,end)
            continue
    t2 = time.time()
    print('\t[used time]: {} seconds.'.format(round(t2-t1, 4)))

def preprocess_input(base_name,text,time_format):
    if '\r\n' in text:
        print(datetime.now().strftime(time_format),
              f'[{base_name}] Found a CRLF -> replace it w/ a space')
        text = text.replace('\r\n', ' ')

    if '\n' in text:
        print(datetime.now().strftime(time_format),
              f'[{base_name}] Found a line break -> replace it w/ a space')
        text = text.replace('\n', ' ')

    if '\t' in text:
        print(datetime.now().strftime(time_format),
              f'[{base_name}] Found a tab -> replace w/ a space')
        text = text.replace('\t', ' ')

    if '\xa0' in text:
        print(datetime.now().strftime(time_format),
              f'[{base_name}] Found a \\xa0 -> replace w/ a space')
        text = text.replace('\xa0', ' ')

    if '\x0b' in text:
        print(datetime.now().strftime(time_format),
              f'[{base_name}] Found a \\x0b -> replace w/ a space')
        text = text.replace('\x0b', ' ')
        
    if '\x0c' in text:
        print(datetime.now().strftime(time_format),
              f'[{base_name}] Found a \\x0c -> replace w/ a space')
        text = text.replace('\x0c', ' ')
    if '|' in text:
        text = text.replace('|', '/')
    
    # remove non-ascii
    text = text.encode("ascii", "ignore").decode()

    found_too_long_words = 0
    max_word_len=50
    tokens=CoNLL_tokenizer(text)
    for idx, tk in enumerate(tokens):
        if len(tk) > max_word_len:
            tokens[idx] = tk[:max_word_len]
            found_too_long_words += 1
    if found_too_long_words > 0:
        print(datetime.now().strftime(time_format),
              f'[{base_name}] Found a too long word -> cut the suffix of the word')
        text = ' '.join(tokens)

    return text

def medline2pubtator(case,save_path):
    time_format='[%d/%b/%Y %H:%M:%S.%f]'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    medline_save_path=f'./case/{case}/medline/'
    for file in os.listdir(medline_save_path):
        if case not in file:
            continue
        base_name=file.replace('.txt','')
        cur_path=os.path.join(medline_save_path, file)
        pubtator_save_path=os.path.join(save_path, base_name+'.PubTator')
        with open(cur_path) as handle,open(pubtator_save_path,'w',encoding='utf-8') as fw:
            records = Medline.parse(handle)
            for record in records:
                pmid=record['PMID']
                if 'TI' not in record and 'AB' not in record:
                    continue
                title=''
                abstract=''
                if 'TI' in record:
                    title=record['TI']
                    title=preprocess_input(base_name,title,time_format)
                if 'AB' in record:
                    abstract=record['AB']
                    abstract=preprocess_input(base_name,abstract,time_format)
                fw.write(pmid+'|t|'+title+'\n')
                fw.write(pmid+'|a|'+abstract+'\n')
                fw.write('\n')
        os.remove(cur_path)
    print("Convert to PubTator format!")


def main():
    # parse parameters
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--case',help='Specify case name')
    argparser.add_argument('-t', '--term',action='store_true',help='Search query for literature retrieval')
    argparser.add_argument('-p', '--pmid',action='store_true',help='PubMed ID(s) to fetch specific papers')
    args = argparser.parse_args()
    if not args.case:
        argparser.print_help()
        print("\n" + "="*60)
        print("❌ ERROR: You must provide the name of case!")
        print("="*60)
        sys.exit(1)
    if not args.term and not args.pmid:
        argparser.print_help()
        print("\n" + "="*60)
        print("❌ ERROR: You must provide at least one input source!")
        print("   Use -t for search term OR -p for PMIDs OR both")
        print("="*60)
        sys.exit(1)
    case=args.case
    pmids=[]
    if args.pmid:
        pmid_file = os.path.join(f'./case/{case}', f'{case}.pmid.txt')
        with open(pmid_file,encoding='utf-8') as fp:
            pmids+=fp.read().strip().split('\n')
    
    if args.term:
        term_file = os.path.join(f'./case/{case}', f'{case}.term.txt')
        term2pmid_files=os.path.join(f'./case/{case}', f'{case}.term2pmids.txt')
        if os.path.exists(term2pmid_files):
            with open(term2pmid_files,encoding='utf-8') as fp:
                pmids+=fp.read().strip().split('\n')
        else:
            pmids+=term_to_pmid(term_file, save_path=term2pmid_files)
    pmids=list(set(pmids))
    

    medline_save_path=f'./case/{case}/medline/'
    pubtator_save_path=f'./case/{case}/pubtator/'
    download_medline(case,pmids,medline_save_path)
    medline2pubtator(case,pubtator_save_path)
    

if __name__ == "__main__":
    main()

