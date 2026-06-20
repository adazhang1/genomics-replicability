FROM nvidia/cuda:13.3.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_OVERRIDE_CUDA=13.3

RUN apt-get update && \
    apt-get install -y wget bzip2 gcc && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda update -n base -c defaults conda -y

COPY prespecified.yml assessment_prespecified.yml
COPY models/basenji/prespecified.yml basenji_prespecified.yml
COPY models/enformer/prespecified.yml enformer_prespecified.yml
COPY GSEA_tissue_cancer_error/prespecified.yml gsea_prespecified.yml

RUN conda init
RUN conda env create -f assessment_prespecified.yml && rm assessment_prespecified.yml
RUN conda env create -f basenji_prespecified.yml && rm basenji_prespecified.yml
RUN conda env create -f enformer_prespecified.yml && rm enformer_prespecified.yml
RUN conda env create -f gsea_prespecified.yml && rm gsea_prespecified.yml
RUN conda clean -afy

CMD ["bash"]
