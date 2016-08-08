#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 25 Jul 2016
# Last modified : 25 Jul 2016

"""
Convert utterance label files from AUD and put them into corresponding docs.
"""

import os 
import sys 
import pickle
import argparse 

def read_simple_flist(fname):

    lst = []
    with open(fname, 'r') as fpr:
        lst = fpr.read().strip().split("\n")
    if lst[-1].strip() == "":
        lst = lst[:-1]
    return lst


def read_label_file(lab_f):
    dec_str = ""
    with open(lab_f, 'r') as fpr:
        for line in fpr:
            dec_str += line.strip().split()[-1] + " "
    return dec_str.strip()


def main():
    """ main method """
    
    pwd = os.path.dirname(os.path.realpath(__file__)) + "/"
    in_lab_dir = os.path.realpath(args.in_lab_dir) + "/"

    model = in_lab_dir.split("/")[-2]  # get model name
    corpus = pwd.split("/")[-3]

    eval_d = pwd + "../../../../../EVAL/topics/" + corpus + "/" + model + "/"

    out_txt_dir = os.path.realpath(args.out_txt_dir) + "/"
    lab_ext = args.lab_ext

    if os.path.exists(out_txt_dir) is False:
        os.makedirs(out_txt_dir, exist_ok=True)
        print(out_txt_dir, "created.")

    all_utt_f = pwd + "../data/all_unique.keys"

    train_doc_f = pwd + "../data/training_docs.keys"
    test_doc_f = pwd + "../data/test_docs.keys"

    all_utt_ids = read_simple_flist(all_utt_f)
    all_doc_ids = read_simple_flist(train_doc_f)
    all_doc_ids += read_simple_flist(test_doc_f)

    utt2doc = pickle.load(open(pwd + "../data/utt2doc.pkl", "rb"))

    in_lab_files = os.listdir(in_lab_dir)
    lab_ids = [lf.split(".")[0] for lf in in_lab_files if lf.split(".")[-1] == lab_ext]

    doc_data = {}
    for lid in lab_ids:
        doc_id = utt2doc[lid]
        lab_f = in_lab_dir + lid + "." + lab_ext
        lab_str = read_label_file(lab_f)
        
        try:
            doc_data[doc_id].append(lab_str)
        except KeyError:
            doc_data[doc_id] = [lab_str]

    for doc_id, data in doc_data.items():
        with open(out_txt_dir + doc_id + ".txt", "w") as fpw:
            fpw.write("\n".join(data) + "\n")
    print(len(doc_data), "docs saved in", out_txt_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_lab_dir", help="AUD label dir")
    # parser.add_argument("out_txt_dir", help="output txt dir")
    parser.add_argument("--lab_ext", "-le", default="lab",
                        help="label ext (default=lab)")
    
    args = parser.parse_args()
    
    main()
