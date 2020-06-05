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

cd ${CDIR}

# binarize
file_path=korean/all.txt
tokenizer_name=pytorch.all.bpe.4.8m_step 
dump_file=korean/binarized_text

function binarize {
python distillation/scripts/binarized_data.py \
    --file_path      ${file_path} \
    --tokenizer_type bert \
    --tokenizer_name ${tokenizer_name} \
    --dump_file      ${dump_file}
}

# count occurences
data_file=korean/binarized_text.${tokenizer_name}.pickle
counts_dump=korean/token_counts.${tokenizer_name}.pickle
vocab_size=100102

function token_counts {
python distillation/scripts/token_counts.py \
    --data_file         ${data_file} \
    --token_counts_dump ${counts_dump} \
    --vocab_size        ${vocab_size}
}

# distillation
student_config=distilbert-base.json
dump_path=korean/kor-distil-bpe-bert

function train {
python distillation/train.py \
    --student_type   distilbert \
    --student_config ${student_config} \
    --teacher_type   bert \
    --teacher_name   ${tokenizer_name} \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path      ${dump_path}  \
    --data_file      ${data_file} \
    --token_counts   ${counts_dump} \
    --force # overwrites the `dump_path` if it already exists.
}

binarize
token_counts
train
