#!/bin/bash

set -o nounset
set -o errexit

VERBOSE_MODE=0

function error_handler()
{
  local STATUS=${1:-1}
  [ ${VERBOSE_MODE} == 0 ] && exit ${STATUS}
  echo "Exits abnormally at line "`caller 0`
  exit ${STATUS}
}
trap "error_handler" ERR

PROGNAME=`basename ${BASH_SOURCE}`
DRY_RUN_MODE=0

function print_usage_and_exit()
{
  set +x
  local STATUS=$1
  echo "Usage: ${PROGNAME} [-v] [-v] [--dry-run] [-h] [--help]"
  echo ""
  echo " Options -"
  echo "  -v                 enables verbose mode 1"
  echo "  -v -v              enables verbose mode 2"
  echo "      --dry-run      show what would have been dumped"
  echo "  -h, --help         shows this help message"
  exit ${STATUS:-0}
}

function debug()
{
  if [ "$VERBOSE_MODE" != 0 ]; then
    echo $@
  fi
}

#GETOPT=`getopt -o vh --long dry-run,help -n "${PROGNAME}" -- "$@"`
GETOPT=`getopt vh $*`
if [ $? != 0 ] ; then print_usage_and_exit 1; fi

eval set -- "${GETOPT}"

while true
do case "$1" in
     -v)            let VERBOSE_MODE+=1; shift;;
     --dry-run)     DRY_RUN_MODE=1; shift;;
     -h|--help)     print_usage_and_exit 0;;
     --)            shift; break;;
     *) echo "Internal error!"; exit 1;;
   esac
done

if (( VERBOSE_MODE > 1 )); then
  set -x
fi

if [ ${#} != 0 ]; then print_usage_and_exit 1; fi

set -o errexit
function readlink()
{
    TARGET_FILE=$2
    cd `dirname $TARGET_FILE`
    TARGET_FILE=`basename $TARGET_FILE`

    # Iterate down a (possible) chain of symlinks
    while [ -L "$TARGET_FILE" ]
    do
        TARGET_FILE=`readlink $TARGET_FILE`
        cd `dirname $TARGET_FILE`
        TARGET_FILE=`basename $TARGET_FILE`
    done

    # Compute the canonicalized name by finding the physical path
    # for the directory we're in and appending the target file.
    PHYS_DIR=`pwd -P`
    RESULT=$PHYS_DIR/$TARGET_FILE
    echo $RESULT
}
export -f readlink

# current dir of this script
CDIR=$(readlink -f $(dirname $(readlink -f ${BASH_SOURCE[0]})))
PDIR=$(readlink -f $(dirname $(readlink -f ${BASH_SOURCE[0]}))/..)


DATA_DIR=${CDIR}/wikitext-2-raw
FILE_SUFFIX=raw
VOCAB_SIZE=50265
TOKENIZER_NAME=roberta

TRAIN_FILE=${DATA_DIR}/wiki.train.raw
EVAL_FILE=${DATA_DIR}/wiki.test.raw
OUTPUT_DIR=${CDIR}/roberta-base.v1
MODEL_TYPE=roberta
CONFIG_DIR=${CDIR}/config-roberta-base

function train_tokenizer {
  python train-tokenizer.py --data_dir=${DATA_DIR} --file_suffix=${FILE_SUFFIX} --vocab_size=${VOCAB_SIZE} --min_frequency=2 --tokenizer_name=${TOKENIZER_NAME}
  mv ${CDIR}/${TOKENIZER_NAME}-vocab.json ${CONFIG_DIR}/vocab.json
  mv ${CDIR}/${TOKENIZER_NAME}-merges.txt ${CONFIG_DIR}/merges.txt
}

function train_lm {
  python ${CDIR}/run_language_modeling.py \
    --output_dir ${OUTPUT_DIR} \
    --model_type ${MODEL_TYPE} \
    --do_train \
    --train_data_file=${TRAIN_FILE} \
    --do_eval \
    --eval_data_file=${EVAL_FILE} \
    --mlm \
    --config_name ${CONFIG_DIR} \
    --tokenizer_name ${CONFIG_DIR} \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_gpu_train_batch_size 16 \
    --evaluate_during_training \
    --seed 42 
}

rm -rf ${OUTPUT_DIR}

export CUDA_VISIBLE_DEVICES=0

train_tokenizer
train_lm
