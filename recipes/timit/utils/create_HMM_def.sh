#!/usr/bin/env bash
#same version as MakeMyHMMproto.sh, but you can set everything on command line
# do.MyHMMproto.sh phoneme_list output_hmmdefs


#dict=$WORK_DIR/dict/dict
#hmmdefs_file=$WORK_DIR/HMM/40gauss.2

if [ $# -ne 2 ]; then
    echo $0 '<in_phonemes> <out_hmmdefs.hvite>'
    exit 1
fi

dict=$1
hmmdefs_file=$2

C=1.0

##########################################################
#make hmmdefs

#number of phonemes
num_phn=`wc $dict | awk '{print $1}'`

zero=""
for((i=0;$i<$num_phn;i++));do
  zero="$zero 0"
done


echo " ~o <VecSize> $num_phn <USER> " > $hmmdefs_file

count=0;
for phn in `cat $dict`; do
  tmp=""
  for((i=0;$i<$num_phn;i++));do
    if [ $i -eq $count ]; then
      tmp="$tmp $C"
    else
      tmp="$tmp 1e30"
    fi
  done
  count=$((count + 1))
  tmp_phn=`echo $phn | sed 's/\(.*\)__[0-9]/\1/'`


    echo "~s \"$phn\"
    <Mean> $num_phn
      $zero
    <Variance> $num_phn
      $tmp 
    <GConst> 0">> $hmmdefs_file

done


for phn in `cat $dict | sed 's/\(.*\)__[0-9]/\1/'|sort|uniq`; do
echo "~h \"$phn\"
 <BeginHMM>
   <NumStates> 5
   <State> 2 ~s \"${phn}__1\"
   <State> 3 ~s \"${phn}__2\"
   <State> 4 ~s \"${phn}__3\"
   <TransP> 5
      0  1   0   0   0
      0  0.5 0.5 0   0
      0  0   0.5 0.5 0
      0  0   0   0.5 0.5
      0  0   0   0   0
   <EndHMM>" >> $hmmdefs_file
done


