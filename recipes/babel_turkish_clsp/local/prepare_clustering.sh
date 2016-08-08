#!/bin/bash 
if [ $# -ne 1 ] ; then
  echo usage: $0 "<setup.sh>"
  exit 1
fi

source $1 || exit 1

echo "---------------------------------------------" 
echo "Creating Word Transcriptions for Clustering"
echo "---------------------------------------------" 
echo "Train"
./local/babel2cluster.py $path_to_train_ref $eval_path/train/word_transcriptions
echo "Dev"
./local/babel2cluster.py $path_to_dev_ref $eval_path/dev/word_transcriptions

echo "-------------------------------------------------------------------" 
echo "Creating Phone Transcriptions from 1rst Pronunciation for Clustering"
echo "-------------------------------------------------------------------" 
echo "Train"
./local/words2phones.py $eval_path/train/word_transcriptions $eval_path/train/phone_transcriptions $path_to_lexicon
echo "Dev"
./local/words2phones.py $eval_path/dev/word_transcriptions $eval_path/dev/phone_transcriptions $path_to_lexicon

echo "-----------------------------------------------------------"
echo " Creating Context Phone Transcriptions from Best Aligned Pronunciation"
echo "------------------------------------------------------------"
echo "Train"
./local/babel2cluster.py $path_to_train_ali $eval_path/train/ali_context_phone_transcriptions
echo "Dev"
./local/babel2cluster.py $path_to_dev_ali $eval_path/dev/ali_context_phone_transcriptions

echo "-----------------------------------------------------------"
echo " Creating Phone Transcriptions from Best Aligned Pronunciation"
echo "------------------------------------------------------------"
echo "Train"
./local/words2phones.py $eval_path/train/ali_context_phone_transcriptions $eval_path/train/ali_phone_transcriptions $path_to_phones True
echo "Dev"
./local/words2phones.py $eval_path/dev/ali_context_phone_transcriptions $eval_path/dev/ali_phone_transcriptions $path_to_phones True

echo "---------------------------------------------------------" 
echo "Creating Decoded Context Phone Transcriptions for Clustering"
echo "---------------------------------------------------------"
echo "Train"
./local/babel2cluster.py $path_to_train_decode $eval_path/train/decoded_context_phone_transcriptions
echo "Dev"
./local/babel2cluster.py $path_to_dev_decode $eval_path/dev/decoded_context_phone_transcriptions

echo "-----------------------------------------------------" 
echo "Creating Decoded Phone Transcriptions for Clustering"
echo "------------------------------------------------------"
echo "Train"
./local/words2phones.py $eval_path/train/decoded_context_phone_transcriptions $eval_path/train/decoded_phone_transcriptions $path_to_phones True
echo "Dev"
./local/words2phones.py $eval_path/dev/decoded_context_phone_transcriptions $eval_path/dev/decoded_phone_transcriptions $path_to_phones True

if $morfessor ; then
  echo "----------------------------------------" 
  echo "Creating Morphemic Transcriptions"
  echo "----------------------------------------"
  if [ ! -d $eval_path/morfessor ] ; then
    mkdir -p $eval_path/morfessor
  fi
  
  command -v morfessor || (echo "Morfessor not installed" && exit 1)
  
  echo "Fetching all words in training and dev and removing * and empty lines."
  cat $eval_path/{train,dev}/word_transcriptions/* | grep -oE '[a-zA-Z<>\*]+' | sort -f | sed 's/*//g' | sed '/^$/d' >  $eval_path/morfessor/morfessor_words.txt
  
  echo "Creating Morfessor test data by removing duplicate words for testing."
  sort -u -f $eval_path/morfessor/morfessor_words.txt > $eval_path/morfessor/words.txt
  echo "Training morfessor."
  morfessor-train -S $eval_path/morfessor/morphs_model.txt $eval_path/morfessor/morfessor_words.txt
  
  echo "Segmenting words."
  morfessor-segment -L $eval_path/morfessor/morphs_model.txt $eval_path/morfessor/words.txt > $eval_path/morfessor/morphs.txt
  
  echo "Creating morphemic lexicon."
  paste -d ' ' $eval_path/morfessor/words.txt $eval_path/morfessor/morphs.txt > $eval_path/morfessor/lexicon_morphemic.txt
 
  echo "Train" 
  ./local/words2morphs.py $eval_path/train/word_transcriptions $eval_path/train/morph_transcriptions $eval_path/morfessor/lexicon_morphemic.txt
  echo "Dev"
  ./local/words2morphs.py $eval_path/dev/word_transcriptions $eval_path/dev/morph_transcriptions $eval_path/morfessor/lexicon_morphemic.txt
fi
# Make final directories for data dump
mkdir -p $eval_path/{words,phonemes,ali_context_phonemes,ali_phonemes,asr_context_phonemes,asr_phonemes}
ln -s $eval_path/train/word_transcriptions/* $eval_path/words/
ln -s $eval_path/dev/word_transcriptions/* $eval_path/words/

ln -s $eval_path/train/phone_transcriptions/* $eval_path/phonemes/
ln -s $eval_path/dev/phone_transcriptions/* $eval_path/phonemes/

ln -s $eval_path/train/ali_context_phone_transcriptions/* $eval_path/ali_context_phonemes
ln -s $eval_path/dev/ali_context_phone_transcriptions/* $eval_path/ali_context_phonemes

ln -s $eval_path/train/ali_phone_transcriptions/* $eval_path/ali_phonemes
ln -s $eval_path/dev/ali_phone_transcriptions/* $eval_path/ali_phonemes

ln -s $eval_path/train/decoded_context_phone_transcriptions/* $eval_path/asr_context_phonemes/
ln -s $eval_path/dev/decoded_context_phone_transcriptions/* $eval_path/asr_context_phonemes/

ln -s $eval_path/train/decoded_phone_transcriptions/* $eval_path/asr_phonemes/
ln -s $eval_path/dev/decoded_phone_transcriptions/* $eval_path/asr_phonemes/


if $morfessor ; then
  mkdir -p $eval_path/morphemes
  ln -s $eval_path/train/morph_transcriptions/* $eval_path/morphs/
  ln -s $eval_path/dev/morph_transcriptions/* $eval_path/morphs/
fi

echo "Clustering preparation seems to have worked"
exit 0
