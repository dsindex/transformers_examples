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

MAX_LENGTH=180
MODEL_NAME_OR_PATH=engeval-model
OUTPUT_DIR=engeval-model
SEED=1

function evaluate {
  local _OUTPUT_DIR=$1

  python ${CDIR}/token-classification/run_ner.py \
      --data_dir ${CDIR}/data \
      --labels ${CDIR}/data/labels.txt \
      --model_name_or_path ${MODEL_NAME_OR_PATH} \
      --output_dir ${_OUTPUT_DIR} \
      --max_seq_length  ${MAX_LENGTH} \
      --seed ${SEED} \
      --do_eval \
      --do_predict
}

evaluate ${OUTPUT_DIR}
