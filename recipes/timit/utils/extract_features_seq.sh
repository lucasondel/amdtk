setup=$1

source $setup || exit 1

mkdir -p $fea_dir
while read line
do
  audio=`echo $line | cut -d':' -f1`
  key=`echo $line | cut -d':' -f2`
  ./utils/extract_features.sh $setup $audio $key
done < $scp
