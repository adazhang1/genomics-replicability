# Differential GSEA: rank genes by (Cancer - Not Cancer), then run fgsea.
# Input : data/normalized_{Basenji,Enformer}_cancer.csv
#         (must contain "Cancer" and "Not Cancer" columns)
# Output: results/differential_gsea/<basename>_differential_GSEA.{csv,rds}

# Locate scripts/ directory regardless of launch method (Rscript or source())
.script_dir = local({
  args = commandArgs(trailingOnly = FALSE)
  fa   = grep("^--file=", args, value = TRUE)
  if (length(fa)) return(normalizePath(dirname(sub("^--file=", "", fa[1]))))
  for (f in rev(sys.frames())) if (!is.null(f$ofile)) return(normalizePath(dirname(f$ofile)))
  normalizePath(".")
})
source(file.path(.script_dir, "_common.R"))

OUT_DIR = file.path(ROOT, "results", "differential_gsea")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

run_differential = function(input_file, gene_id_ds, pathways, out_dir) {
  prefix = sub("\\.csv$", "", basename(input_file))
  message("[", prefix, "] differential GSEA (Cancer - Not Cancer)")

  dt = fread(input_file)
  dt$differential = dt$Cancer - dt$`Not Cancer`
  dt$ENSG = gsub("\\..+", "", dt$ENSG)

  dt_mapped = inner_join(gene_id_ds[, c(1, 2)], dt,
                         by = c("ensembl_gene_id" = "ENSG"))
  stats = dt_mapped$differential
  names(stats) = as.character(dt_mapped$entrezgene_id)
  stats = stats[!is.na(stats) & !is.infinite(stats)]

  res = fgsea(pathways = pathways, stats = stats,
              minSize = 5, maxSize = 500) %>%
    arrange(pval) %>%
    dplyr::select(-leadingEdge)

  saveRDS(res, file.path(out_dir, paste0(prefix, "_differential_GSEA.rds")))
  fwrite(res, file.path(out_dir, paste0(prefix, "_differential_GSEA.csv")))
  invisible(res)
}

message("Loading GO pathways ...")
pathways   = load_go_pathways()
gene_id_ds = build_gene_id_map()
message("Loaded ", length(pathways), " pathways; mapped ", nrow(gene_id_ds), " genes")

inputs = list.files(DATA_DIR,
                    pattern = "^normalized_(Basenji|Enformer)_cancer\\.csv$",
                    full.names = TRUE)
message("Processing ", length(inputs), " input files")
for (f in inputs) run_differential(f, gene_id_ds, pathways, OUT_DIR)
message("Done.")
