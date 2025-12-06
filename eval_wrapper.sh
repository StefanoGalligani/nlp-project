#!/bin/bash

if [ $# -lt 3 ]; then
    echo "<predictions.json> <sa_dev sa_test_v sa_test_p or sp_dev or sp_test> <out_base_dir: results/> [nj: 4]"
    exit;
fi

pred=$1
dset=$2
out_base=$3
mkdir -p "${3}"
nj=${4:-4}

# Get script directory for reliable paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval_src="${SCRIPT_DIR}/evaluate.py"
fuzzy_src="${SCRIPT_DIR}/preprocessor_fuzzy.py"

python="python3"

gold_pre="${SCRIPT_DIR}/jga_reference"

if [ "${dset}" == "sa_dev" ]; then
    gold="${gold_pre}/sa_multiwoz_dev_jga_reference.json"
    db="sa"
elif [ "${dset}" == "sa_test_v" ]; then
    gold="${gold_pre}/sa_multiwoz_test_verbatim_jga_reference.json"
    db="sa"
elif [ "${dset}" == "sa_test_p" ]; then
    gold="${gold_pre}/sa_multiwoz_test_paraphrased_jga_reference.json"
    db="sa"
elif [ "${dset}" == "sp_dev" ]; then
    gold="${gold_pre}/spokenwoz_dev_jga_reference.json"
    db="sp"
elif [ "${dset}" == "sp_test" ]; then
    gold="${gold_pre}/spokenwoz_test_jga_reference.json"
    db="sp"
else
    echo "${dset} not understood."
    exit;
fi

base=$(echo "${pred}" | awk -F'/' '{print $(NF-1)}')
fname=$(echo "${pred}" | awk -F'/' '{print $(NF)}' | sed 's/.json//g')
# echo $base
# echo $fname

out_score="${out_base}/${base}/${fname}_score.json"
mkdir -p "${out_base}/${base}"

${python} ${eval_src} \
    --dst \
    --golden "${gold}" \
    --input "${pred}" \
    --output "${out_score}"

echo "-- FUZZY --"

out_fuzzy_pred="${out_base}/${base}/${fname}_fuzzy.json"
out_fuzzy_score="${out_base}/${base}/${fname}_fuzzy_score.json"

echo "${out_fuzzy_pred}"

${python} ${fuzzy_src} --in_json "${pred}" "--${db}" --out_json "${out_fuzzy_pred}" --nj ${nj}

${python} ${eval_src} \
    --dst \
    --golden "${gold}" \
    --input "${out_fuzzy_pred}" \
    --output "${out_fuzzy_score}"