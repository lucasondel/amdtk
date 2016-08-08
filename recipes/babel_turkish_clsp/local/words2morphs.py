#! /usr/bin/python

import sys
import glob
import codecs
import os

def main():
  INPUT_DIR = sys.argv[1]
  OUTPUT_PHONES = sys.argv[2]
  LEXICON = sys.argv[3]

  if(not os.path.exists(OUTPUT_PHONES)):
    os.makedirs(OUTPUT_PHONES)

  w2p = lexicon_map(LEXICON)
  word_files = glob.glob("%s/*.txt" % INPUT_DIR)
  for wf_i,wf in enumerate(word_files):
    sys.stdout.write("File %d of %d \r" % (wf_i+1,len(word_files)))
    sys.stdout.flush()
    utt_id = os.path.basename(wf)
    with codecs.open(wf,"r","utf-8") as fp:
      with codecs.open(OUTPUT_PHONES + "/" + utt_id,"w","utf-8") as fpo:
        for line in fp:
          words = line.strip().split(" ")
          # Catch the oov case
          try:
            phone_transcription = " ".join([w2p[w] for w in words])
          except:
            phone_transcription = ""
            for w in words:
              try:
                phone_transcription += w2p[w] + " "
              except:
                for symb in "*-~"
                  w = w.replace(symb,"")
                if(w in w2p.keys()):
                  phone_transcription += w2p[w] + " "
                else:
                  phone_transcription += w + " "

          fpo.write("%s\n" % phone_transcription)
  print("")
                   
def lexicon_map(lexicon):
  w2p = {}
  with codecs.open(lexicon,"r","utf-8") as fp:
    last_line = ""
    for line in fp:
      line_vals = line.strip().split(" ")
      # Take only first key (pronunciation from kaldi format lexicon)
      if(line_vals[0] != last_line):
        w2p[line_vals[0]] = " ".join(line_vals[1:])
        last_line = line_vals[0] 
  return w2p

if __name__ == "__main__":
  main()
