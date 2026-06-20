#!/usr/bin/env bash

#
# Run the full Genomics Replicability analysis
#

USE_DOCKER=${USE_DOCKER:-false}
DOCKER_TAG="genomics-replicability"
ROOT_DIR="$(pwd)"

if [ "$USE_DOCKER" = "true" ]; then
    CMD_PREFIX=(docker run --gpus=all -v "${ROOT_DIR}:/home/app" -w /home/app "localhost/${DOCKER_TAG}:latest")
else
    CMD_PREFIX=()
fi

MAIN_ENV="performance_assessment"
BASENJI_ENV="basenji"
ENFORMER_ENV="enformer"
GSEA_ENV="GSEA_tissue_cancer"

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
    docker build . -t "${DOCKER_TAG}"
else
    conda env create -f prespecified.yml
    conda env create -f models/basenji/prespecified.yml
    conda env create -f models/enformer/prespecified.yml
    conda env create -f GSEA_tissue_cancer_error/prespecified.yml
    conda clean -afy
fi

## Run GSEA Analysis
cd "${ROOT_DIR}/GSEA_tissue_cancer_error" || exit

echo "Starting Per Sample GSEA..."
"${CMD_PREFIX[@]}" conda run -n "${GSEA_ENV}" Rscript scripts/01_per_sample_gsea.R

echo "Starting differential GSEA"
"${CMD_PREFIX[@]}" conda run -n "${GSEA_ENV}" Rscript scripts/02_differential_gsea.R

echo "Finished; Summarizing results (See GSEA_tissue_cancer_error/explore_results.ipynb)"
# Note: Can replace `--to notebook` with `--to pdf` or `--to html` for outputs
# that don't depend on a running jupyter instance.
"${CMD_PREFIX[@]}" conda run -n "${GSEA_ENV}" jupyter nbconvert \
		   --export \
		   --to notebook \
		   --inplace explore_results.ipynb

## Train Basenji model
cd "${ROOT_DIR}/models/basenji/" || exit

# NOTE for Ada: I know there are other runs that need to be done if you can
# fill those out.
echo "Training Basenji..."
"${CMD_PREFIX[@]}" conda run -n "${BASENJI_ENV}" python basenji_train.py \
		   original/params.json "${BASENJI_DATA_DIR}"

## Making predictions
cd "${ROOT_DIR}/models/basenji" || exit

echo "Making predictions with Basenji..."
"{CMD_PREFIX[@]}" conda run -n "${BASENJI_ENV}" python predict_test_set.py \
		  original "${BASENJI_DATA_DIR}"

cd "${ROOT_DIR}/models/enformer" || exit

echo "Making predictions with Enformer..."
"{CMD_PREFIX[@]}" conda run -n "${ENFORMER_ENV}" python \
		  predict_test_set.py "${BASENJI_DATA_DIR}"

## Performing top level analysis of results
cd "${ROOT_DIR}" || exit

echo "Performing main analysis..."
"{CMD_PREFIX[@]}" conda run -n "${MAIN_ENV}" jupyter nbconvert \
		  --export \
		  --to notebook \
		  --inplace Main.ipynb
