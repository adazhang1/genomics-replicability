# Shared setup: locate project root, load pathway DB, build gene ID map.
# Sourced by 01_per_sample_gsea.R and 02_differential_gsea.R.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(fgsea)
  library(data.table)
})

# Resolve ROOT = parent of scripts/ regardless of how the script is launched
.find_root = function() {
  args = commandArgs(trailingOnly = FALSE)
  file_arg = grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(file.path(dirname(sub("^--file=", "", file_arg[1])), "..")))
  }
  if (!is.null(sys.frames()) && length(sys.frames()) > 0) {
    for (f in rev(sys.frames())) {
      ofile = f$ofile
      if (!is.null(ofile)) {
        return(normalizePath(file.path(dirname(ofile), "..")))
      }
    }
  }
  normalizePath("..")
}
ROOT = .find_root()

DATA_DIR = file.path(ROOT, "data")
RES_DIR  = file.path(ROOT, "resources")

load_go_pathways = function(res_dir = RES_DIR) {
  pathways = list()
  for (f in c("MSigDB-2025.1.Hs-C5_GO.Biological.Process.RData",
              "MSigDB-2025.1.Hs-C5_GO.Cellular.Component.RData",
              "MSigDB-2025.1.Hs-C5_GO.Molecular.Function.RData")) {
    load(file.path(res_dir, f))
    for (k in seq_len(ncol(concept.mat$mat))) {
      pathways[[colnames(concept.mat$mat)[k]]] =
        rownames(concept.mat$mat)[concept.mat$mat[, k]]
    }
  }
  pathways
}

# ENSG -> Entrez map restricted to genes present in gene_list.txt
build_gene_id_map = function(data_dir = DATA_DIR, res_dir = RES_DIR) {
  gene_id = fread(file.path(data_dir, "gene_list.txt"), header = FALSE)$V1 %>%
    gsub("\\..+", "", .)

  ds = readRDS(file.path(res_dir, "Ensembl_Gene_110_GRCh38.p14_20230913.rds")) %>%
    dplyr::select(ensembl_gene_id:external_gene_name) %>%
    drop_na() %>%
    unique() %>%
    filter(ensembl_gene_id %in% gene_id) %>%
    arrange(entrezgene_id) %>%
    filter(!duplicated(ensembl_gene_id)) %>%
    filter(!duplicated(entrezgene_id))

  indx = match(gene_id, ds$ensembl_gene_id)
  ds[indx[!is.na(indx)], ]
}
