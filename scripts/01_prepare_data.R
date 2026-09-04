#!/usr/bin/env Rscript
# S1 - Data curation.
#
# Converts the published benchmark release into the arrays the Python pipeline reads,
# builds the four relation graphs, and asserts the QC conditions that the rest of the
# project depends on. Run once.
#
# Inputs (downloaded by scripts/00_download.py):
#   double_perturbation_results_predictions.RDS   Zenodo 10.5281/zenodo.16092690
#   double_perturbation_results_parameters.RDS
#   data/raw/dorothea_hs.rda, data/raw/tftargets.rda
#
# Outputs:
#   data/splits/split_{1..5}.json         the published splits, verbatim
#   data/processed/split_{1..5}.npz       Y, A, per-method predictions (via TSV -> npz)
#   data/graphs/regulatory_matrix.csv     TF-target pair scores
#   data/graphs/profilesim_split{s}.csv   effect-profile similarity, train singles only
#   data/graphs/perturbed_genes.txt
suppressPackageStartupMessages({library(jsonlite)})
# Run from the repository root. The benchmark RDS files are expected in the working
# directory (scripts/00_download.py puts them there).
D <- "."
stopifnot(dir.exists(file.path(D, "data")))

pr <- readRDS("double_perturbation_results_predictions.RDS")
pa <- readRDS("double_perturbation_results_parameters.RDS")
nm <- sapply(pa, function(e) e$name)
get <- function(meth, split) pr[[which(nm == paste0("norman_from_scfoundation-", split, "-", meth))]]
meths <- sort(unique(sub("^.*-\\d+-", "", nm)))
meths_pred <- setdiff(meths, c("ground_truth", "additive_model"))

# --- canonical perturbation labels -------------------------------------------------
# Three methods label pairs "AHR_FEV" while ground truth uses "AHR+FEV"; without this
# the label sets do not intersect at all and every prediction silently becomes NA.
canon <- function(x) {
  sapply(as.character(x), function(s) {
    if (is.na(s) || s == "") return(NA_character_)
    p <- unlist(strsplit(s, "[+_]")); p <- p[p != "" & p != "ctrl"]
    if (length(p) == 0) return(NA_character_)
    if (length(p) == 1) return(paste0(p, "+ctrl"))
    paste(sort(p), collapse = "+")
  }, USE.NAMES = FALSE)
}

gt <- get("ground_truth", 1); ad <- get("additive_model", 1)
genes <- names(gt$prediction[[1]])
gt_c <- canon(gt$perturbation)
ref <- gt_c[!is.na(gt_c)]                       # 224 perturbations (one control-only label dropped)
cat("reference perturbations:", length(ref), " genes:", length(genes), "\n")

# --- QC1: gene ordering identical across every prediction vector -------------------
stopifnot(all(sapply(gt$perturbation, function(k) identical(names(gt$prediction[[k]]), genes))))
cat("QC1 gene ordering identical: TRUE\n")

# --- gene subset: top-variance genes in the RESIDUAL on split-1 TRAIN doubles ------
Yf <- sapply(gt$perturbation, function(k) gt$prediction[[k]])
Af <- sapply(ad$perturbation, function(k) ad$prediction[[k]])
colnames(Yf) <- gt_c; colnames(Af) <- canon(ad$perturbation)
e1 <- pa[[which(nm == "norman_from_scfoundation-1-additive_model")]]
tr1 <- canon(e1$test_train_labels$train); tr1 <- tr1[!is.na(tr1)]
tr_d <- intersect(tr1, ref[!grepl("\\+ctrl$", ref)])
gv <- rowMeans((Yf[, tr_d, drop = FALSE] - Af[, tr_d, drop = FALSE])^2)
NG <- 2000
sel <- sort(order(gv, decreasing = TRUE)[1:NG])
cat("gene subset:", NG, "genes,", signif(100 * sum(gv[sel]) / sum(gv), 2),
    "% of training residual SS\n")
dir.create(file.path(D, "data/processed"), showWarnings = FALSE, recursive = TRUE)
writeLines(genes[sel], file.path(D, "data/processed/genes_selected.txt"))

# --- QC2: the additive model is exact on singles -----------------------------------
sing <- grep("\\+ctrl$", ref, value = TRUE)
rms_s <- sqrt(mean((Yf[sel, sing] - Af[sel, sing])^2))
cat("QC2 additive RMS on", length(sing), "singles:", signif(rms_s, 4), "\n")
stopifnot(rms_s < 1e-8)

# --- splits, verbatim --------------------------------------------------------------
dir.create(file.path(D, "data/splits"), showWarnings = FALSE, recursive = TRUE)
for (s in 1:5) {
  e <- pa[[which(nm == paste0("norman_from_scfoundation-", s, "-additive_model"))]]
  L <- e$test_train_labels
  cl <- function(v) { x <- canon(v); x[!is.na(x)] }
  tr <- cl(L$train); te <- cl(L$test)
  stopifnot(length(intersect(tr, te)) == 0)      # QC3
  write_json(list(split = s, config_id = e$parameters$test_train_config_id,
                  dataset = e$parameters$dataset_name,
                  train = tr, test = te, val = cl(L$val)),
             file.path(D, sprintf("data/splits/split_%d.json", s)), auto_unbox = TRUE)
}
cat("QC3 train/test disjoint in all 5 splits: TRUE\n")

# --- per-split matrices ------------------------------------------------------------
fill <- function(e) {
  ec <- canon(e$perturbation)
  M <- matrix(NA_real_, length(sel), length(ref), dimnames = list(NULL, ref))
  for (k in seq_along(ec))
    if (!is.na(ec[k]) && ec[k] %in% ref && !is.null(e$prediction[[k]]))
      M[, ec[k]] <- e$prediction[[k]][sel]
  M
}
for (s in 1:5) {
  write.table(fill(get("ground_truth", s)), file.path(D, sprintf("data/processed/Y_split%d.tsv", s)),
              sep = "\t", quote = FALSE, col.names = NA)
  write.table(fill(get("additive_model", s)), file.path(D, sprintf("data/processed/A_split%d.tsv", s)),
              sep = "\t", quote = FALSE, col.names = NA)
  for (m in meths_pred)
    write.table(fill(get(m, s)), file.path(D, sprintf("data/processed/pred_%s_split%d.tsv", m, s)),
                sep = "\t", quote = FALSE, col.names = NA)
  cat("split", s, "matrices written\n")
}

# --- relation graphs ---------------------------------------------------------------
pg <- sub("\\+ctrl$", "", sing)
dir.create(file.path(D, "data/graphs"), showWarnings = FALSE, recursive = TRUE)
writeLines(pg, file.path(D, "data/graphs/perturbed_genes.txt"))

e_dor <- new.env(); load(file.path(D, "data/raw/dorothea_hs.rda"), envir = e_dor)
dor <- e_dor$dorothea_hs
allreg <- split(dor$target, dor$tf)
e_tft <- new.env(); load(file.path(D, "data/raw/tftargets.rda"), envir = e_tft)
flat <- function(x) if (is.list(x[[1]])) {
  u <- list(); for (ct in x) for (tf in names(ct)) u[[tf]] <- unique(c(u[[tf]], ct[[tf]])); u
} else x
for (src in c("ENCODE", "ITFP", "Neph2012", "TRED", "TRRUST", "Marbach2016")) {
  x <- flat(e_tft[[src]])
  for (tf in names(x)) allreg[[tf]] <- unique(c(allreg[[tf]], as.character(x[[tf]])))
}
cat("regulons: ", length(allreg), " TFs; ", sum(pg %in% names(allreg)),
    "/100 perturbed genes have one\n", sep = "")

jacc <- function(a, b) { if (!length(a) || !length(b)) return(0)
                         i <- length(intersect(a, b)); i / (length(a) + length(b) - i) }
np <- length(pg)
REG <- matrix(0, np, np, dimnames = list(pg, pg))
for (i in seq_len(np)) for (j in seq_len(np)) if (i < j) {
  a <- allreg[[pg[i]]]; b <- allreg[[pg[j]]]
  d <- (!is.null(a) && pg[j] %in% a) || (!is.null(b) && pg[i] %in% b)
  REG[i, j] <- REG[j, i] <- max(jacc(a, b), if (d) 0.5 else 0)
}
ut <- upper.tri(REG)
cat("regulatory graph: ", sum(REG[ut] > 0), "/", sum(ut), " pairs (",
    round(100 * mean(REG[ut] > 0)), "%)\n", sep = "")
write.csv(REG, file.path(D, "data/graphs/regulatory_matrix.csv"))

# effect-profile similarity, per split, from TRAINING singles only.
# All 100 singles are training data in every published split (only doubles are held
# out), so this is leak-free by construction; the per-split loop keeps it that way if
# the splits ever change.
S1 <- Yf[sel, sing, drop = FALSE]; colnames(S1) <- pg
for (s in 1:5) {
  sp <- fromJSON(file.path(D, sprintf("data/splits/split_%d.json", s)))
  tr_g <- sub("\\+ctrl$", "", intersect(sing, sp$train))
  M <- matrix(0, np, np, dimnames = list(pg, pg))
  if (length(tr_g) > 2) M[tr_g, tr_g] <- cor(S1[, tr_g, drop = FALSE])
  diag(M) <- 0
  write.csv(M, file.path(D, sprintf("data/graphs/profilesim_split%d.csv", s)))
}
cat("profile-similarity graphs written for 5 splits\n")
cat("\nNow run: python scripts/01b_to_npz.py  (TSV -> npz)  and  python scripts/01c_string.py\n")
