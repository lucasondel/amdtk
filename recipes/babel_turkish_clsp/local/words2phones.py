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

  inverse_lexicon = False
  if(len(sys.argv[0:]) >= 5):
    inverse_lexicon = True 

  w2p = lexicon_map(LEXICON,inverse=inverse_lexicon)
  word_files = glob.glob("%s/*.txt" % INPUT_DIR)
  word_count = 0.0
  oov_count = 0.0
  for wf_i,wf in enumerate(word_files):
    sys.stdout.write("File %d of %d \r" % (wf_i+1,len(word_files)))
    sys.stdout.flush()
    utt_id = os.path.basename(wf)
    with codecs.open(wf,"r","utf-8") as fp:
      with codecs.open(OUTPUT_PHONES + "/" + utt_id,"w","utf-8") as fpo:
        for line in fp:
          words = line.strip().split(" ")
          # Catch the oov case
          word_count += len(words) 
          try:
            phone_transcription = " ".join([w2p[w] for w in words])
          except:
            phone_transcription = ""
            for w in words:
              try:
                phone_transcription += w2p[w] + " "
              except:  
                # Catch the mispronounced words or stutters
                for symb in "*-~":
                  w = w.replace(symb,"")
                
                if( w in w2p.keys()):
                  phone_transcription += w2p[w] + " "
                else:
                  oov_count += 1
                  phone_transcription += "<oov> "
          
          fpo.write("%s\n" % phone_transcription)
    
  print("") 
  print("OOV RATE: %f" % (oov_count/word_count))
  print("OOVs = %d, Words = %d" %(oov_count,word_count))
               
def lexicon_map(lexicon,inverse=False):
  w2p = {}
  with codecs.open(lexicon,"r","utf-8") as fp:
    last_line = ""
    for line in fp:
      line_vals = line.strip().split(" ")
      if(not inverse):
        # Take only first key (pronunciation from kaldi format lexicon)
        if(line_vals[0] != last_line):
          w2p[line_vals[0]] = " ".join(line_vals[1:])
          last_line = line_vals[0]
      else:
        # Take only the first key (pronunciation from kaldi format lexicon)
        if(line_vals[1] != last_line):
          w2p[line_vals[1]] = line_vals[0]
          last_line = line_vals[0]
  return w2p

if __name__ == "__main__":
  main()
