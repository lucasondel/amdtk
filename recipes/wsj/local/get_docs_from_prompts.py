#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 08 Jul 2016
# Last modified : 11 Jul 2016

"""------------------------------------------------------------------------
Browse the WSJ0 and WSJ1 corpora, parse the prompts (transcriptions),
and categorize the utts into documents (documents are articles).
This article info is obtained from the <PROMPT_ID>.

Dumps the unique train/test keys in data/
------------------------------------------------------------------------"""

import os 
import sys
# import socket
import string
import argparse
import re


def read_lexicon(lex_file):
    """ Read the lexicon and load it in dictionary """

    lex = {}
    phs = {}
    with open(lex_file, "r") as fpr:
        for line in fpr:
            line = line.strip().lower()
            tokens = line.split(" ")
            ph_seq = ""
            for i, tok in enumerate(tokens):
                if tok.strip() == '':
                    continue
                else:
                    if i > 0:
                        ph = re.sub('[0-9]', '', tok)
                        phs[tok] = ph
                        ph_seq += ph + " "
            lex[tokens[0]] = ph_seq.strip()

    if VERBOSE:
        print('No. of phonemes:', len(phs), 'after mapping:',
              len(set(list(phs.values()))))
        print('No. of words in lexicon:', len(lex))

    return lex

                            
def read_all_prompts(fpaths):
    """ Get all prompts in one list """

    data = []
    for fp in fpaths:
        with open(fp, 'r') as fpr:
            data += fpr.read().split("\n")

    new_data = [d for d in data if len(d.strip()) > 1]
    return new_data


def read_simple_flist(fpath):
    """ read every line and put in list """
    
    fids = []
    with open(fpath, 'r') as fpr:
        fids = fpr.read().split("\n")

    if fids[-1].strip() == '':
        fids = fids[:-1]
    return fids


def get_ptx_fpaths(out_fpath_file):
    """ Get all the file paths of prompts files """

    os.system("find " + WSJ0 + " -type f -name \"*.ptx\" > " + out_fpath_file)
    os.system("find " + WSJ1 + " -type f -name \"*.ptx\" >> " + out_fpath_file)

    
def get_docIDs_from_prompts(data, doc_d, utt_d, utt_txt_d, utt_ph_d, lex):
    """ Parse the prompts and get the utt to doc ID mappings """

    found = 0
    not_found = 0  # tokens not found in lexicon
    incom = 0

    txt_utt_d = {}  # utt txt to utt ID mapping (to get the unique utts)
    # oth = {}
    for utt_line in data:
        utt_line = utt_line.strip()
        vals = utt_line.split("(")

        id_tmp = vals[-1][:-1]
        utt = utt_line[:-len(id_tmp)-2].strip().lower()

        """
        translator = str.maketrans({key: None for key in string.punctuation})
        clean_utt = utt.translate(translator)
        clean_utt = re.sub("\s\s+", " ", clean_utt)  # remove multiple spaces
        utt = clean_utt
        """
        utt = re.sub("\.|,|\"|\?|\(|\)|;|\&|\$|\%|\{|\}|\[|\]|:|/|~|`|\!", "", utt)
        utt = re.sub("\-", " ", utt)

        # m = re.search("^\'[a-z]", utt)
        # if m is not None:
        #    utt = re.sub("\'", "", utt)
        
        pt_tmp = id_tmp.split(" ")[-1].split(".")
        utt_id = id_tmp.split(" ")[0].strip()
        
        # https://catalog.ldc.upenn.edu/docs/LDC93S6A/csrnov92.html
        # ptx format (<UTT_ID> <PROMPT_ID>)
        # PROMPT_ID = <YEAR>.<FILE-NUMBER>.<ARTICLE-NUMBER>.<PARAGRAPH-NUMBER>.<SENTENCE-NUMBER>
        
        # article ID as doc ID
        
        doc_id = ''
        if len(pt_tmp) == 5:
            doc_id = pt_tmp[2]  # 2 => get article ID
        else:            
            incom += 1
            # oth[pt_tmp[0]] = 1
            
        # update the doc_d dictionary
        if doc_id in doc_d:
            doc_d[doc_id].append(utt_id)
        else:
            doc_d[doc_id] = [utt_id]

        # check if the sentence is repeating
        if utt in txt_utt_d:
            txt_utt_d[utt].append(utt_id)
        else:
            txt_utt_d[utt] = [utt_id]

        # update the utt_d and utt_txt_d dictionaries
        if utt_id in utt_d:
            continue
        else:
            utt_d[utt_id] = doc_id
            utt_txt_d[utt_id] = utt

            utt_ph = ""
            tokens = utt.split()
            for tok in tokens:
                try:
                    utt_ph += lex[tok] + " "
                    found += 1
                except KeyError:
                    not_found += 1
                    # m = re.search('[0-9]+', tok)                    
                    # if m is None:
                    #    print(tok) #, 'not found in lexicon.')
                    
            utt_ph_d[utt_id] = utt_ph
            
    if VERBOSE:
        print('Utts with incomplete prompt IDs:', incom)
        print('No. of tokens not found in lexicon:', not_found,
              '({:.2f} %)'.format((float(not_found) * 100) / found))

    return txt_utt_d

        
def dump_utts_into_docs(utt_ids, doc_d, utt_txt_d, utt_ph_d, out_word_dir,
                        out_ph_dir, txt_utt_d, pwd, base):
    """ Dump the utts in utt_ids into corresponding documents and save them
    in out_word_dir/ out_ph_dir/"""

    fpu = None
    if VERBOSE:
        fpu = open(pwd + '../data/repeating_utts_' + base + '.txt', 'w')
    
    count = 0
    uniq_utt = {}
    uniq_keys = []
    uniq_doc_ids = []

    for doc_id in sorted(list(doc_d.keys())):
        utt_l = sorted(doc_d[doc_id])
        out_word_f = out_word_dir + doc_id + ".txt"
        out_ph_f = out_ph_dir + doc_id + ".txt"

        utts_to_dump = []
        utts_to_dump_ph = []

        utt_l2 = sorted(list(set(utt_ids) & set(utt_l)))
        count += len(utt_l2)
        for utt_id in utt_l2:

            try:
                utt_ids_l = txt_utt_d[utt_txt_d[utt_id]]
                if VERBOSE:
                    if len(utt_ids_l) > 0:
                        for uid in utt_ids_l:
                            fpu.write(utt_txt_d[utt_id] + ":" + uid + "\n")                    

            except KeyError:
                print('Cannot find sentence.')

            try:
                uniq_utt[utt_txt_d[utt_id]] += 1
            except KeyError:
                uniq_utt[utt_txt_d[utt_id]] = 1
                # utts_to_dump.append(utt_id + " " + utt_txt_d[utt_id])
                # utts_to_dump_ph.append(utt_id + " " + utt_ph_d[utt_id])
                utts_to_dump.append(utt_txt_d[utt_id])
                utts_to_dump_ph.append(utt_ph_d[utt_id])
                uniq_keys.append(utt_id)

        if len(utts_to_dump) > 0:
            uniq_doc_ids.append(doc_id)
            with open(out_word_f, 'w') as fpw, open(out_ph_f, 'w') as fpp:
                fpw.write("\n".join(utts_to_dump) + "\n")
                fpp.write("\n".join(utts_to_dump_ph) + "\n")                              

    uniq_keys = sorted(uniq_keys)
    uniq_key_f = pwd + "../data/" + base + "_unique.keys"
    with open(uniq_key_f, 'w') as fpw:
        fpw.write("\n".join(uniq_keys) + "\n")

    uniq_doc_ids = sorted(uniq_doc_ids)
    uniq_doc_f = pwd + "../data/" + base + "_docs.keys"
    with open(uniq_doc_f, 'w') as fpw:
        fpw.write("\n".join(uniq_doc_ids) + "\n")
    
    if VERBOSE:
        print("No. of utts used:", count)
        print("No. of unique utts:", len(uniq_utt))
        fpu.close()

              
def main():
    """ main method """

    pwd = os.path.dirname(os.path.realpath(__file__)) + "/"
    out_fpath_file = pwd + "../data/prompts.fpaths"
    get_ptx_fpaths(out_fpath_file)

    fpaths = read_simple_flist(out_fpath_file)

    lex_file = pwd + "../data/lexicon.txt"
    lex = read_lexicon(lex_file)
    
    if VERBOSE:
        print('Total no. of prompt files:', len(fpaths))
    
    # data = read_ptx_file('all_ptx.txt')
    data = read_all_prompts(fpaths)

    if VERBOSE:
        print('Total no. of prompts:', len(data))
    
    utt_txt_d = {}  # utt ID to text mapping
    utt_ph_d = {}  # utt ID to phoneme seq mapping
    utt_d = {}  # utt ID to docID mapping
    doc_d = {}  # doc to utt [] mapping

    txt_utt_d = get_docIDs_from_prompts(data, doc_d, utt_d, utt_txt_d,
                                        utt_ph_d, lex)

    if VERBOSE:
        with open(pwd + '../data/unique_utt_IDs.txt', 'w') as fpw:
            for txt, uid_l in txt_utt_d.items():
                fpw.write(txt + " " + ",".join(uid_l) + "\n")

        print('No. of docs (articles):', len(doc_d))   
        print('No. of utts with doc IDs:', len(utt_d))
        print('No. of utts with doc IDs and text:', len(utt_txt_d))
        print('No. of unique utts (based on text):', len(txt_utt_d))

    train_ids = read_simple_flist(pwd + '../data/training_si84.keys')
    train_ids += read_simple_flist(pwd + '../data/training_si284.keys')

    test_ids = read_simple_flist(pwd + '../data/test_eval92.keys')

    if VERBOSE:
        print('Train utt IDs:', len(train_ids))
        print('Test utt IDs:', len(test_ids))

    # Dump the utts in respective documents
    out_dir = os.path.realpath(pwd + "../../../../../../EVAL/topics/wsj/") + "/"
    train_out = out_dir + "words/"
    test_out = out_dir + "words/"

    train_ph_out = out_dir + "phonemes/"
    test_ph_out = out_dir + "phonemes/"
    
    os.makedirs(out_dir, exist_ok=True)
    os.system("mkdir -p " + train_out + " " + test_out + " " + \
              train_ph_out + " " + test_ph_out)

    if VERBOSE:
        print('Created the dirs:')
        print(out_dir + '\n' + train_out + '\n' + test_out + '\n' + \
              train_ph_out + '\n' + test_ph_out)
        
    dump_utts_into_docs(sorted(train_ids), doc_d, utt_txt_d, utt_ph_d, 
                        train_out, train_ph_out, txt_utt_d, pwd, "training")
    dump_utts_into_docs(sorted(test_ids), doc_d, utt_txt_d, utt_ph_d, 
                        test_out, test_ph_out, txt_utt_d, pwd, "test")
    
    print('Data preparation for topic based document clustering is done.')
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wsj0', help='path to wsj0')
    parser.add_argument('wsj1', help='path to wsj1')

    parser.add_argument('--verbose', action='store_true',
                        help='Display useless information while processing.')
    args = parser.parse_args()

    WSJ0 = os.path.realpath(args.wsj0) + "/"
    WSJ1 = os.path.realpath(args.wsj1) + "/"
    VERBOSE = args.verbose
    
    main()
                
    """
    Lucas didn't like this automation

    host_addr = socket.gethostbyaddr(socket.gethostname())[0]
    host = host_addr.split(".")[1]

    VERBOSE = False
    WSJ0 = ''
    WSJ1 = ''
    
    # BUT cluster
    if host == "fit":
        print('Host: BUT cluster')
        WSJ0 = "/mnt/matylda2/data/WSJ0/"
        WSJ1 = "/mnt/matylda2/data/WSJ1/"
        
    # CLSP cluster
    elif host == "clsp":
        print('Host: CLSP cluster')
        WSJ0 = "/export/corpora5/LDC/LDC93S6B/"
        WSJ1 = "/export/corpora5/LDC/LDC94S13B/"

    else:        
        print("Manually enter the path of WSJ0 and WSJ1 in the source file:",
              sys.argv[0])
        sys.exit()
    """ 
