# Per-sample GSEA for each normalized stat table.
# Input : data/normalized_{Basenji,Enformer}_{cancer,tissue,tissue_and_cancer}.csv
# Output: results/per_sample_gsea/<basename>_per_sample_GSEA.rds
#
# Each input CSV has columns: ENSG, <gene_metadata>..., <sample_1>, <sample_2>, ...
# Sample columns start at column 4 after the inner_join with the gene ID map.

# Locate scripts/ directory regardless of launch method (Rscript or source())
.script_dir = local({
  args = commandArgs(trailingOnly = FALSE)
  fa   = grep("^--file=", args, value = TRUE)
  if (length(fa)) return(normalizePath(dirname(sub("^--file=", "", fa[1]))))
  for (f in rev(sys.frames())) if (!is.null(f$ofile)) return(normalizePath(dirname(f$ofile)))
  normalizePath(".")
})
source(file.path(.script_dir, "_common.R"))

OUT_DIR = file.path(ROOT, "results", "per_sample_gsea")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

gsea_per_sample = function(input_stat, gene_id_ds, pathways, prefix, out_dir) {
  input_stat$ENSG = gsub("\\..+", "", input_stat$ENSG)
  input_stat = inner_join(gene_id_ds[, c(1, 2)], input_stat,
                          by = c("ensembl_gene_id" = "ENSG"))

  lapply(4:ncol(input_stat), function(i) {
    # Resumable: skip per-sample shards already on disk
    shard = file.path(out_dir, paste0(prefix, "_sample_", i - 3, "_GSEA.rds"))
    if (file.exists(shard)) return(readRDS(shard))

    stats = input_stat[[i]]
    names(stats) = as.character(input_stat$entrezgene_id)
    if (anyNA(stats) || any(is.infinite(stats))) {
      message("  skip sample ", i - 3, " (NA/Inf)")
      return(NULL)
    }

    # fgsea's adaptive multilevel branch occasionally hits an internal
    # split.data.table bug (missing modeFraction/denomProb cols); the bug is
    # stochastic, so retry with a fresh RNG seed before giving up.
    res = NULL
    for (attempt in seq_len(5)) {
      set.seed(1000L * (i - 3) + attempt)
      res = tryCatch(
        fgsea(pathways = pathways, stats = stats, minSize = 5, maxSize = 500),
        error = function(e) {
          message("  fgsea attempt ", attempt, " failed on sample ", i - 3,
                  ": ", conditionMessage(e))
          NULL
        }
      )
      if (!is.null(res)) break
    }
    if (is.null(res)) {
      message("  skip sample ", i - 3, " (fgsea failed after retries)")
      return(NULL)
    }
    res = res %>%
      arrange(pval) %>%
      mutate(tissue = colnames(input_stat)[i]) %>%
      dplyr::select(-leadingEdge)

    saveRDS(res, shard)
    res
  }) %>% bind_rows()
}

run_one = function(input_file, gene_id_ds, pathways, out_dir) {
  prefix = sub("\\.csv$", "", basename(input_file))
  message("[", prefix, "] per-sample GSEA")

  input_stat = fread(input_file)
  res = gsea_per_sample(input_stat, gene_id_ds, pathways, prefix, out_dir)
  saveRDS(res, file.path(out_dir, paste0(prefix, "_per_sample_GSEA.rds")))

  # Drop intermediate per-sample shards once the combined RDS is written
  shards = list.files(out_dir,
                      pattern = paste0("^", prefix, "_sample_.*_GSEA\\.rds$"),
                      full.names = TRUE)
  if (length(shards)) file.remove(shards)
  invisible(res)
}

message("Loading GO pathways ...")
pathways   = load_go_pathways()
gene_id_ds = build_gene_id_map()
message("Loaded ", length(pathways), " pathways; mapped ", nrow(gene_id_ds), " genes")

inputs = list.files(DATA_DIR,
                    pattern = "^normalized_(Basenji|Enformer)_.*\\.csv$",
                    full.names = TRUE)
message("Processing ", length(inputs), " input files")
for (f in inputs) run_one(f, gene_id_ds, pathways, OUT_DIR)
message("Done.")
