#! /usr/bin/python

import sys
import codecs
import os

def main():
  if(len(sys.argv[1:]) < 2 or not sys.argv[1] or not sys.argv[2]):
    sys.stderr.write("Bad commandline arguments")
    sys.exit(1)

  INPUT  = sys.argv[1] 
  OUTPUT = sys.argv[2]
  
  if( not os.path.exists(OUTPUT)):
    os.makedirs(OUTPUT)  

  utt_ids = []
  with codecs.open(INPUT,"r","utf-8") as fp:
    # Initialize utt_id to none
    utt_id_curr = None
    
    # For each line write to a new file for each new utt_id
    for line in fp:
      
      # Retrieve transcription and utterance id 
      line_parts = line.strip().split(" ")
      utt_id_new = "_".join(line_parts[0].split("_")[0:4])
      utt_transcription = " ".join(line_parts[1:])
      if(utt_id_new != utt_id_curr):
        utt_ids.append(utt_id_new)
        if(utt_id_curr):
          fpo.close()
        fpo = codecs.open(OUTPUT + "/" + utt_id_new + ".txt", "w", "utf-8")     
      fpo.write("%s\n" % utt_transcription)
      utt_id_curr = utt_id_new
    fpo.close()
  
  # Write utterance files
  with open(OUTPUT + "/../docs.txt","w") as fpo:
    for u in utt_ids:
      fpo.write("%s.txt\n" % u) 

if __name__ == "__main__":
  main() 
       
    
