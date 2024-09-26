# Archaea vs Bacteria

Code from paper:

>Systematic genome-guided discovery of antagonistic interactions between archaea and bacteria
>
>Romain Strock, Valerie WC Soo, Antoine Hocher, Tobias Warnecke<br>
>bioRxiv 2024.09.18.613068; doi: https://doi.org/10.1101/2024.09.18.613068


## Content

- `notebook/`: python notebooks to reproduce all the figures.
- `src/`: scripts to produce all the data used in the paper. Scripts are referenced in notebooks.
- `figures/`: all unassembled figures as produced by the relevant notebooks.
- `data/`: input data and data produced by the scripts and notebooks in this repo.

## Install

```sh
git clone https://github.com/srom/archaea-vs-bacteria.git
git lfs pull origin main  # optional - see below
cd archaea-vs-bacteria
conda env create -f environment.yml
conda activate AvB
```

Note: some of the larger files in `data/` are tracked with [Git Large File Storage](https://git-lfs.com/). They won't get  downloaded with `git pull` or through GitHub's UI. To download these files, install `git-lfs` and use `git lfs pull`.
