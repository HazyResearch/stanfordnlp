#!/bin/bash
#
# Train and evaluate parser. Run as:
#   ./run_depparse.sh TREEBANK TAG_TYPE OTHER_ARGS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and OTHER_ARGS are additional training arguments (see parser code) or empty.
# This script assumes UDBASE and DEPPARSE_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
save_name=$1; shift
sample_train=$1; shift
args=$@
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

train_file=./data/conllu/ud-treebanks-v2.3/${treebank}/${short}-ud-train.conllu
eval_file=./data/conllu/ud-treebanks-v2.3/${treebank}/${short}-ud-dev.conllu
output_file=${DEPPARSE_DATA_DIR}/${short}.dev.pred.conllu
gold_file=./data/conllu/ud-treebanks-v2.3/${treebank}/${short}-ud-dev.conllu

# train_file=./data/conllu/ud-treebanks-v2.3/${treebank}/${short}-ud-train.conllu
# eval_file=${DEPPARSE_DATA_DIR}/en_ewt.dev.in1.conllu
# output_file=${DEPPARSE_DATA_DIR}/${short}.dev.pred.conllu
# gold_file=${DEPPARSE_DATA_DIR}/en_ewt.dev.in1.conllu

# test_eval_file =./data/conllu/ud-treebanks-v2.3/${treebank}/${short}-ud-test.conllu
# test_output_file=${DEPPARSE_DATA_DIR}/${short}.test.pred.conllu
# test_gold_file=./data/conllu/ud-treebanks-v2.3/${treebank}/${short}-ud-test.conllu


if [ ! -e $train_file ]; then
    echo "In the if statement for train file"
    bash scripts/prep_depparse_data.sh $treebank $tag_type
fi

batch_size=1


echo "Using batch size $batch_size"
echo "Running parser with $args..."


python -m stanfordnlp.models.parser --wordvec_dir $WORDVEC_DIR --train_file $train_file --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --batch_size $batch_size \
    --sample_train $sample_train --save_name $save_name\
    --mode train $args
python -m stanfordnlp.models.parser --wordvec_dir $WORDVEC_DIR --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --save_name $save_name \
    --mode predict $args

# python -m stanfordnlp.models.parser --wordvec_dir $WORDVEC_DIR --eval_file $test_eval_file \
#     --output_file $test_output_file --gold_file $test_gold_file --lang $lang --shorthand $short --save_name $save_name \
#     --mode predict $args



results=`python stanfordnlp/utils/conll18_ud_eval.py -v $gold_file $output_file | head -12 | tail -n+12 | awk '{print $7}'`
echo $results $args >> ${DEPPARSE_DATA_DIR}/${short}.results
echo $short $results $args
