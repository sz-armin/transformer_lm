#!/bin/bash

# Test data
FPATH="data/dev.txt"
# sentencepeace model
MPATH="/home/is/armin-sa/Projects/lm/data/spm_dec/spm_dec.model"

sed -e 's/^$/▮/' $FPATH > tmp.txt

spm_encode --model=$MPATH --output_format=id --extra_options=eos < tmp.txt > test_id_tmp.txt
rm tmp.txt

tr '\n' ' ' < test_id_tmp.txt > test_id.txt
sed -i -e 's/267 1 2/1/g' test_id.txt
rm test_id_tmp.txt