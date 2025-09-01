# Archaea vs Bacteria

Code from paper: 

> Archaea produce peptidoglycan hydrolases that kill bacteria
>
> Romain Strock, Valerie WC Soo, Pauline Misson, Georgia Roumelioti, Pavel V Shliaha, Antoine Hocher, Tobias Warnecke<br>
> PLOS Biology 23(8): e3003235; doi: https://doi.org/10.1371/journal.pbio.3003235

## Content

- `notebook/`: python notebooks to reproduce all the figures.
- `src/`: scripts to produce all the data used in the paper. Scripts are referenced in notebooks.
- `figures/`: all unassembled figures as produced by the relevant notebooks.
- `data/`: input data and data produced by the scripts and notebooks in this repo.
  - includes supplementary data such as tree alignments or the result of homology search.

## Install

```sh
git clone https://github.com/srom/archaea-vs-bacteria.git
git lfs pull origin main  # optional - see below
cd archaea-vs-bacteria
conda env create -f environment.yml
conda activate AvB
```

Note: some of the larger files in `data/` are tracked with [Git Large File Storage](https://git-lfs.com/). They won't get  downloaded with `git pull` or through GitHub's UI. To download these files, install `git-lfs` and use `git lfs pull`.
