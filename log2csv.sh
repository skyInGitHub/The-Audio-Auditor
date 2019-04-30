#!/bin/bash

## Convert xxx.log to xxx.csv
##
## Step1: xxx.log to xxx.txt ---> reduce some rebundent info
## Step2: select strings from xxx.txt to .csv
## Step2: formalize the xxx.csv with features


# Get input strings from cmd
# if ["$#" -ne 1]; then
#   echo "Usage: $0 <path/to/log> <path/to/new_csv>"
#   echo "e.g: $0 data/train/shd2_in.log data/train/train_in_2.csv"
#   exit 1
# fi

# log=$1
#csv=$2
log="shd1_out.log"

# log2txt
tmp="tmp.txt"
sed -e 'h;s/.*utterance //' $log > $tmp
sed -i '1d' $tmp
for ((i=1;i<=3;i++)); do
  sed -i '$d' $tmp
done

#sed -i :a -e '$d;N;2,3ba' -e 'P;D' -i $tmp

#sed -i '${lines}s/ /,/1' < input.txt