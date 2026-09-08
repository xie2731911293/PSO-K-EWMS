# PSO-K-EWMS — public code

Accompanying the manuscript

> *PSO-K-EWMS: Hypergraph-Enhanced Adaptive Clustering for High-Dimensional Data*
> (Applied Intelligence, manuscript APIN-D-26-03970R1)

## Files

| file | description |
|---|---|
| `pso_k_ewms.py` | PSO-K-EWMS reference implementation (self-contained; includes the EWMS base clustering it builds on). See the file header/docstring for the algorithm steps and equation references. |
| `CVAE.py` | CVAE-based gene screening used for the grape double-cropping data. Command-line configurable; conditions on the one-hot encodings of `Season`, `GrowthStage` and `cultivar`. |

## Dependencies

```
numpy
scipy
pandas
scikit-learn
torch
```

## Usage

- `pso_k_ewms.py` can be run standalone (it includes a small self-contained
  example). Details, defaults and random-seed handling are given in its
  docstring.
- `CVAE.py` expects a gene-expression spreadsheet (columns `Sample`, `Season`,
  `GrowthStage`, `cultivar`, then one column per gene). Example:

  ```bash
  python CVAE.py --input meta_gene_common.xlsx --epochs 100 --seed 42
  python CVAE.py -h            # list all options
  ```

## Statistics (Nemenyi critical difference)

The recomputed Friedman / Nemenyi analysis used in the manuscript is provided for
reproducibility:

- `figures/significance_critical_difference.png` - the Nemenyi critical-difference
  diagram (18 datasets, 11 methods, mean AMI).
- `analysis/nemenyi_cd_from_table.py` - script that reads Table 4 of the manuscript
  and recomputes the mean ranks, Friedman test and the diagram.
- `analysis/table4_ami_ranks.csv` - the per-dataset AMI values and mean ranks used.

Result summary: PSO-K-EWMS attains the best mean rank (1.06); Friedman
$\chi^2pprox115.1$, $p<10^{-19}$; by the Nemenyi test it is significantly better
than 8 of the 10 baselines, while the two closest methods (EWMS and QS++) are not
declared significantly different.

## Notes

- In this paper the clustering is applied to *objects* given as the rows of the
  data matrix `X`. For the public benchmarks the rows are samples/observations;
  for the grape application the rows are genes, each described by its FPKM
  profile over the RNA-seq samples.
- PSO fitness is the negative mean silhouette of the partition produced for a
  candidate `(h, λ)`; PSO minimises it.
- The 18 public benchmark datasets and the compared baselines follow the curated
  collection and protocol of the reference EWMS study (Kumar et al.,
  *Applied Soft Computing*, 169 (2025) 112347). Remaining experiment-level
  scripts and raw outputs are available from the authors on reasonable request.
