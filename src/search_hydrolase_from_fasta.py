import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Set
import subprocess
import tempfile

import pandas as pd
from Bio import SeqIO
from Bio.SearchIO.HmmerIO.hmmer3_domtab import Hmmer3DomtabHmmqueryParser


logger = logging.getLogger()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_fasta', 
        help='Path to input fasta file', 
        type=str,
        required=True,
    )
    parser.add_argument(
        '-p', '--pfam_hmm', 
        help='Path to Pfam hmm file', 
        type=str,
        required=True,
    )
    parser.add_argument(
        '-o', '--output_path', 
        help='Path to output fasta file', 
        type=str,
        required=True,
    )
    parser.add_argument('--n_cpus', type=int, default=4)
    args = parser.parse_args()

    input_fasta = Path(args.input_fasta)
    pfam_hmm = Path(args.pfam_hmm)
    output_path = Path(args.output_path)
    n_cpus = args.n_cpus

    if not input_fasta.is_file():
        logger.error(f'Input fasta does not exist: {args.input_fasta}')
        sys.exit(1)
    elif not pfam_hmm.is_file():
        logger.error(f'Pfam HMM file does not exist: {args.pfam_hmm}')
        sys.exit(1)

    logger.info(f'Search for Pfam domains from {pfam_hmm} with {n_cpus} CPUs')

    pgh_domains = pd.read_csv(Path(os.getcwd()) / 'data' / 'pgh_domains.csv')
    cw_binding_domains = sorted(pgh_domains[pgh_domains['kind'] == 'Cell wall binding']['short_name'].values)
    catalytic_domains = sorted(pgh_domains[pgh_domains['kind'] == 'Catalytic']['short_name'].values)

    with tempfile.NamedTemporaryFile(suffix='.domtblout.txt', delete=False) as f:
        domtblout_path = Path(f.name).resolve()

    try:
        response = pfam_hmmer_search(input_fasta, pfam_hmm, domtblout_path, n_cpus)

        if response.returncode != 0:
            stderr_txt = response.stderr.decode('utf-8')
            logger.error(f'Error while running `hmmsearch`: {stderr_txt}')
            sys.exit(1)

        logger.info((
            'Identify protein containing at least one catalytic domain and '
            'one cell wall binding domain'
        ))
        pgh_protein_ids = find_pgh_proteins(domtblout_path, catalytic_domains, cw_binding_domains)

    finally:
        if domtblout_path.is_file():
            domtblout_path.unlink()

    logger.info(f'Export PGH proteins to {output_path}')
    export_proteins_matching_ids(pgh_protein_ids, input_fasta, output_path)

    logger.info('DONE')
    sys.exit(0)


def pfam_hmmer_search(
        input_fasta : os.PathLike, 
        pfam_hmm : os.PathLike, 
        domtblout_path : os.PathLike, 
        n_cpus : int,
    ):
    return subprocess.run(
        [
            'hmmsearch',
            '--acc',
            '--noali',
            '-o', '/dev/null',
            '--domtblout', domtblout_path.resolve().as_posix(),
            '--cpu', f'{n_cpus}',
            '--cut_ga',
            pfam_hmm.resolve().as_posix(),
            input_fasta.resolve().as_posix(),
        ], 
        capture_output=True,
    )


def find_pgh_proteins(
    domtblout_path : os.PathLike, 
    catalytic_domains : List[str], 
    cw_binding_domains : List[str],
):
    hmm_data = {
        'protein_id': [],
        'hmm_query': [],
    }
    with domtblout_path.open() as f:
        parser = Hmmer3DomtabHmmqueryParser(f)

        for record in parser:
            for protein_id, hit in record.items:
                hmm_query = hit.query_id
                hmm_data['protein_id'].append(protein_id)
                hmm_data['hmm_query'].append(hmm_query)

    df = pd.DataFrame.from_dict(hmm_data)
    
    hydrolase_subset = df[df['hmm_query'].isin(catalytic_domains)]
    cw_binding_subset = df[df['hmm_query'].isin(cw_binding_domains)]

    protein_set = set(hydrolase_subset['protein_id'].values)
    protein_set &= set(cw_binding_subset['protein_id'].values)
    return protein_set


def export_proteins_matching_ids(
    pgh_protein_ids : Set[str], 
    input_fasta : os.PathLike,
    output_path : os.PathLike,
):
    out_records = []
    with input_fasta.open('r') as f_in:
        for record in SeqIO.parse(f_in, 'fasta'):
            if record.id in pgh_protein_ids:
                out_records.append(record)

    with output_path.open('w') as f_out:
        SeqIO.write(out_records, f_out, 'fasta')


if __name__ == '__main__':
    main()
