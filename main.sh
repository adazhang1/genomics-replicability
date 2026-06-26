#!/usr/bin/env bash

#
# Run the full Genomics Replicability analysis
#

env_name() {
    local fname="$1"

    awk -F': *' '/^name:/ {print $2; exit}' "$fname"
}

run_cmd() {
    if [ "$USE_DOCKER" = "true" ]; then
        docker run --gpus=all -v "$(pwd):/home/app" -w /home/app "localhost/${DOCKER_TAG}:latest" "$@"
    else
        "$@"
    fi
}

USE_DOCKER=${USE_DOCKER:-false}
DOCKER_TAG="genomics-replicability"
ROOT_DIR="$(pwd)"

MAIN_PRESPECIFIED="${ROOT_DIR}/prespecified.yml"
BASENJI_PRESPECIFIED="${ROOT_DIR}/models/basenji/prespecified.yml"
ENFORMER_PRESPECIFIED="${ROOT_DIR}/models/enformer/prespecified.yml"
GSEA_PRESPECIFIED="${ROOT_DIR}/GSEA_tissue_cancer_error/prespecified.yml"

MAIN_ENV="$(env_name "$MAIN_PRESPECIFIED")"
BASENJI_ENV="$(env_name "$BASENJI_PRESPECIFIED")"
ENFORMER_ENV="$(env_name "$ENFORMER_PRESPECIFIED")"
GSEA_ENV="$(env_name "$GSEA_PRESPECIFIED")"

if [ -z "$BASENJI_DATA_DIR" ]; then
    echo "Must set BASENJI_DATA_DIR environment variable"
    exit 1
elif ! [ -d "$BASENJI_DATA_DIR" ]; then
    echo "The value of BASENJI_DATA_DIR (\"${BASENJI_DATA_DIR}\") is not a directory."
    exit 1
fi
BASENJI_DATA_DIR="$(cd "$BASENJI_DATA_DIR" && pwd)"

## Build docker or conda environments
if [ "$USE_DOCKER" = "true" ]; then
    # NOTE: If docker image is outdated, this will need to be manually rebuilt.
    docker image inspect "localhost/${DOCKER_TAG}:latest" >/dev/null 2>&1 ||
        docker build . -t "${DOCKER_TAG}"
else
    conda env create -f "$MAIN_PRESPECIFIED"
    conda env create -f "$BASENJI_PRESPECIFIED"
    conda env create -f "$ENFORMER_PRESPECIFIED"
    conda env create -f "$GSEA_PRESPECIFIED"
    conda clean -afy
fi

## Train Basenji model
cd "${ROOT_DIR}/models/basenji/" || exit

echo "Training Basenji with original parameter file..."
for i in {1..4}; do
    run_cmd conda run -n "${BASENJI_ENV}" python basenji_train.py \
        original/params.json "${BASENJI_DATA_DIR}" -o "original/run_$i"
done

echo "Training Basenji with modified parameter file to include augmentation..."
for i in {1..4}; do
    run_cmd conda run -n "${BASENJI_ENV}" python basenji_train.py \
        augmented/params.json "${BASENJI_DATA_DIR}" -o "original/run_$i"
done

## Making predictions
cd "${ROOT_DIR}/models/basenji" || exit

echo "Making predictions with released (pre-trained) Basenji..."
run_cmd conda run -n "${BASENJI_ENV}" python predict_test_set.py \
    released_model "${BASENJI_DATA_DIR}"

echo "Making predictions with re-trained Basenji models (original param file)..."
for i in {1..4}; do
    run_cmd conda run -n "${BASENJI_ENV}" python predict_test_set.py \
        "original/run_$i" "${BASENJI_DATA_DIR}"
done

echo "Making predictions with re-trained Basenji models (modified param file for augmentation)..."
for i in {1..4}; do
    run_cmd conda run -n "${BASENJI_ENV}" python predict_test_set.py \
        "augmented/run_$i" "${BASENJI_DATA_DIR}"
done

cd "${ROOT_DIR}/models/enformer" || exit

echo "Making predictions with Enformer..."
run_cmd conda run -n "${ENFORMER_ENV}" python \
    predict_test_set.py "${BASENJI_DATA_DIR}"

## Performing top level analysis of results
cd "${ROOT_DIR}" || exit

echo "Performing main analysis..."
run_cmd conda run -n "${MAIN_ENV}" jupyter nbconvert \
    --export \
    --to notebook \
    --inplace Main.ipynb

## Run GSEA Analysis
cd "${ROOT_DIR}/GSEA_tissue_cancer_error" || exit

echo "Starting Per Sample GSEA..."
run_cmd conda run -n "${GSEA_ENV}" Rscript scripts/01_per_sample_gsea.R

echo "Starting differential GSEA"
run_cmd conda run -n "${GSEA_ENV}" Rscript scripts/02_differential_gsea.R

echo "Finished; Summarizing results (See GSEA_tissue_cancer_error/explore_results.ipynb)"
# Note: Can replace `--to notebook` with `--to pdf` or `--to html` for outputs
# that don't depend on a running jupyter instance.
run_cmd conda run -n "${GSEA_ENV}" jupyter nbconvert \
    --export \
    --to notebook \
    --inplace explore_results.ipynb
