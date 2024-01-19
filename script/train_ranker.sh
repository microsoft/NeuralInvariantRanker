# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <codex-model-name[ada, babbage, curie, davinci, ada_002]> <fold> [max_positive] [max_negative] [alpha]"
    exit 1
fi

SCRIPT_PATH="$(dirname -- "${BASH_SOURCE[0]}")" 
BASE_DIR=`realpath $SCRIPT_PATH/../`;
echo "PROJECT BASE DIRECTORY: $BASE_DIR";
export PYTHONPATH=${BASE_DIR}:${PYTHONPATH};

im=$1;
fold=$2;
mp=${3:-"2"};
mn=${4:-"2"};
alpha=${5:-"0"};

name="${im}/mp-${mp}_mn-${mn}_alpha-${alpha}/fold_${fold}"; 

DATA_BASE_DIR="${BASE_DIR}/data/ranker_data";
OUTPUT_DIR="${BASE_DIR}/models/ranker_result/${name}";

CONFIG_FILE="${BASE_DIR}/configs/ranker_config.json";

LOG_DIR="${OUTPUT_DIR}/logs";
mkdir -p ${LOG_DIR};

if [[ $im == "ada" ]]; then
    codex_model="ada-code-search-code"
elif [[ $im == "ada_002" ]]; then
    codex_model="text-embedding-ada-002"
elif [[ $im == "babbage" ]]; then
    codex_model="babbage-code-search-code"
elif [[ $im == "curie" ]]; then
    codex_model="curie-similarity"
elif [[ $im == "davinci" ]]; then
    codex_model="davinci-similarity"
else
    echo "Invalid codex model name: $im"
    exit 1
fi

embedding_path="${DATA_BASE_DIR}/fold_${fold}/embeddings_${im}.json";

if [[ ! -f $embedding_path ]]; then
    echo "Embedding file not found: $embedding_path"
    echo "Please run `python ${DATA_BASE_DIR}/vectorize.py <data_path>` first!"
    exit 1
fi

python $BASE_DIR/src/ranker/main.py \
    --data_path ${DATA_BASE_DIR}/fold_${fold} \
    --embedding_path $embedding_path \
    --training_config ${CONFIG_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --initial_model codex \
    --codex_model $codex_model \
    --max_positive_examples ${mp} \
    --max_negative_examples ${mn} \
    --alpha ${alpha} \
    --do_train \
    --data_cache_path ${OUTPUT_DIR}/data_cache 2>&1| tee ${LOG_DIR}/train_and_evaluate.log;