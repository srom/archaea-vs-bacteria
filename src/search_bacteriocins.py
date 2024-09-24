import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from multiprocessing import Process, Queue
from queue import Empty
import tempfile
from typing import List
import re

import numpy as np
import pandas as pd
from Bio.SearchIO.HmmerIO.hmmer3_domtab import Hmmer3DomtabHmmqueryParser


logger = logging.getLogger()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--assemblies', 
        help='Path to base folder containing assemblies', 
        type=str,
        required=True,
    )
    parser.add_argument(
        '-d', '--db', 
        help='Path to antimicrobial peptides database folder', 
        type=str,
        default='data/bacteriocins_db',
    )
    parser.add_argument(
        '-o', '--output_path', 
        help='Path to output file', 
        type=str,
        required=True,
    )
    parser.add_argument(
        '-e', '--hmmer_e_value', 
        type=float,
        default=1e-6,
        help='Domain E-value threshold (hmmer)', 
    )
    parser.add_argument('--n_processes', type=int, default=2)
    parser.add_argument('--n_threads_per_process', type=int, default=2)

    args = parser.parse_args()
    base_folder = Path(args.assemblies)
    db_folder = Path(args.db)
    output_path = Path(args.output_path)
    hmmer_e_value = args.hmmer_e_value
    n_processes = args.n_processes
    n_threads_per_process = args.n_threads_per_process

    if not base_folder.is_dir():
        logger.error(f'Assemblies folder does not exist: {args.assemblies}')
        sys.exit(1)
    elif not db_folder.is_dir():
        logger.error(f'AMP database does not exist: {args.db}')
        sys.exit(1)

    paths = sorted(
        [
            p for p in base_folder.iterdir()
            if p.is_dir() and p.name.startswith('GC')
        ],
        key=lambda p: p.name
    )
    logger.info(f'Total number of assemblies: {len(paths):,}')
    if len(paths) == 0:
        sys.exit(0)

    n_processes = min(n_processes, len(paths))
    n_per_process = int(np.ceil(len(paths) / n_processes))

    processes = []
    queue = Queue()
    for i in range(n_processes):
        start = i * n_per_process
        end = start + n_per_process
 
        p = Process(target=worker_main, args=(
            i,
            db_folder,
            paths[start:end],
            hmmer_e_value,
            n_threads_per_process,
            queue,
        ))
        p.start()
        processes.append(p)

    partial_paths = []
    for p in processes:
        p.join()
        try:
            temp_path = queue.get_nowait()
            partial_paths.append(temp_path)
        except Empty:
            continue

    # Sort path by worker index (because only first worker contains the csv header)
    partial_paths = sorted(partial_paths, key=lambda t: t[0])

    # Concatenate files
    try:
        with output_path.open('w') as f:
            sorted_paths = [p.as_posix() for _, p in partial_paths]
            returncode = subprocess.call(
                ['cat'] + sorted_paths, 
                stdout=f,
            )
            if returncode != 0:
                logger.error(f'Error while concatenating files')
                sys.exit(1)

    finally:
        for _, p in partial_paths:
            if p.is_file():
                p.unlink()

    # Compress output file
    response = subprocess.run(['gzip', output_path.resolve().as_posix()], capture_output=True)
    if response.returncode != 0:
        stderr_txt = response.stderr.decode('utf-8')
        logger.error(f'Error while compressing CSV output {output_path}: {stderr_txt}')
        sys.exit(1)

    logger.info('DONE')
    sys.exit(0)


def worker_main(
    worker_ix : int, 
    db_folder : os.PathLike, 
    paths : List[os.PathLike], 
    hmmer_e_value : float,
    n_threads_per_process : int,
    queue : Queue,
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    logger.info(f'Worker {worker_ix+1}: STARTING')

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        output_path = Path(f.name).resolve()

    header = worker_ix == 0

    pgh_domains = pd.read_csv(Path(os.getcwd()) / 'data' / 'pgh_domains.csv')
    catalytic_domains = sorted(pgh_domains[pgh_domains['kind'] == 'Catalytic']['short_name'].values)
    cw_binding_domains = sorted(pgh_domains[pgh_domains['kind'] == 'Cell wall binding']['short_name'].values)

    data = []
    save_every = 100
    for i, path in enumerate(paths):
        if i == 0 or (i+1) % save_every == 0 or (i+1) == len(paths):
            logger.info(f'Worker {worker_ix + 1} | Processing assembly {i+1:,} / {len(paths):,}')

        out_df = screen_for_amp(
            path, 
            db_folder,
            hmmer_e_value, 
            n_threads_per_process,
            catalytic_domains,
            cw_binding_domains,
        )

        if len(out_df) > 0:
            data.append(out_df)

        if len(data) >= save_every:
            append_to_output_file(data, output_path, header)
            header = False
            data = []

    if len(data) > 0:
        append_to_output_file(data, output_path, header)

    queue.put((worker_ix, output_path))
    logger.info(f'Worker {worker_ix+1}: DONE')


def screen_for_amp(
    path, 
    db_folder, 
    hmmer_e_value, 
    n_threads_per_process,
    catalytic_domains,
    cw_binding_domains,
):
    assembly_accession = extract_accession_from_path_name(path.name)

    data = {
        'assembly_accession': [],
        'protein_id': [],
        'amp_id': [],
        'amp_desc': [],
        'evalue': [],
        'bitscore': [],
    }

    # Process Pfam
    screen_for_pfam_or_tigr_records('pfam', path, db_folder, assembly_accession, data)

    # Process TIGR
    screen_for_pfam_or_tigr_records('tigr', path, db_folder, assembly_accession, data)

    # Decompress protein fasta file & process HMM and protein files
    protein_path_gz = path / f'{path.name}_protein.faa.gz'
    if not protein_path_gz.is_file():
        logger.error(f'Protein file not found: {protein_path_gz}')
    else:
        with tempfile.NamedTemporaryFile(suffix='.fasta', delete=False) as protein_file:
            protein_path = Path(protein_file.name).resolve()
            returncode = subprocess.call(
                ['gzip', '-cd', protein_path_gz.resolve().as_posix()],
                stdout=protein_file,
            )
        try:
            if returncode != 0:
                logger.error(f'Error while decompressing {protein_path_gz}')
            else:
                # Process PG hydrolase
                screen_for_pg_hydrolases_with_pfam(
                    path, 
                    assembly_accession,
                    data,
                    catalytic_domains,
                    cw_binding_domains,
                )

                # Process HMM
                screen_for_hmm(protein_path, db_folder, assembly_accession, hmmer_e_value, n_threads_per_process, data)

                # Process proteins
                screen_for_proteins(protein_path, db_folder, assembly_accession, n_threads_per_process, data)

        finally:
            if protein_path.is_file():
                protein_path.unlink()

    out_df = pd.DataFrame.from_dict(data)
    out_df['sort_factor'] = out_df['amp_id'].apply(lambda amp_id: 0 if amp_id == 'pg_hydrolase' else 1)
    out_columns = [c for c in out_df.columns if c != 'sort_factor']

    return out_df.sort_values(
        ['assembly_accession', 'protein_id', 'sort_factor', 'evalue']
    ).drop_duplicates(
        ['assembly_accession', 'protein_id']
    )[out_columns]


def screen_for_pg_hydrolases_with_pfam(path, assembly_accession, data, catalytic_domains, cw_binding_domains):
    pfam_path = path / f'{path.name}_Pfam-A.csv.gz'
    pfam_df = pd.read_csv(pfam_path)

    hydrolase_subset = pfam_df[pfam_df['hmm_query'].isin(catalytic_domains)]
    cw_binding_subset = pfam_df[pfam_df['hmm_query'].isin(cw_binding_domains)]

    protein_set = set(hydrolase_subset['protein_id'].values)
    protein_set &= set(cw_binding_subset['protein_id'].values)

    for protein_id in sorted(protein_set):
        hydrolase_domain = sorted(pfam_df[
            (pfam_df['protein_id'] == protein_id) &
            pfam_df['hmm_query'].isin(catalytic_domains)
        ]['hmm_query'].values)[0]

        cw_binding_domain = sorted(pfam_df[
            (pfam_df['protein_id'] == protein_id) &
            pfam_df['hmm_query'].isin(cw_binding_domains)
        ]['hmm_query'].values)[0]

        amp_name = f'{hydrolase_domain}+{cw_binding_domain}'

        hydrolase_domain_row = pfam_df[
            (pfam_df['protein_id'] == protein_id) &
            (pfam_df['hmm_query'] == hydrolase_domain)
        ].iloc[0]
        cw_binding_domain_row = pfam_df[
            (pfam_df['protein_id'] == protein_id) &
            (pfam_df['hmm_query'] == cw_binding_domain)
        ].iloc[0]

        evalue = (hydrolase_domain_row['evalue'] + cw_binding_domain_row['evalue']) / 2
        bitscore = (hydrolase_domain_row['bitscore'] + cw_binding_domain_row['bitscore']) / 2

        data['assembly_accession'].append(assembly_accession)
        data['protein_id'].append(protein_id)
        data['amp_id'].append('pg_hydrolase')
        data['amp_desc'].append(amp_name)
        data['evalue'].append(evalue)
        data['bitscore'].append(bitscore)


def screen_for_pfam_or_tigr_records(identifier, path, db_folder, assembly_accession, data):
    if identifier == 'pfam':
        pfam_path = path / f'{path.name}_Pfam-A.csv.gz'
        df = pd.read_csv(pfam_path)
    else:
        tigr_path = path / f'{path.name}_TIGR.csv.gz'
        df = pd.read_csv(tigr_path)

    amp_df = pd.read_csv(db_folder / f'amp_{identifier}.csv', index_col='hmm_query')

    for query in sorted(amp_df.index):
        amp_row = amp_df.loc[query]
        for row in df[df['hmm_query'] == query].itertuples():
            data['assembly_accession'].append(assembly_accession)
            data['protein_id'].append(row.protein_id)

            if identifier == 'pfam':
                data['amp_id'].append(query)
                data['amp_desc'].append(amp_row['hmm_desc'])
            else:
                data['amp_id'].append(amp_row['hmm_desc'])
                data['amp_desc'].append(query)

            data['evalue'].append(row.evalue)
            data['bitscore'].append(row.bitscore)


def screen_for_hmm(protein_path, db_folder, assembly_accession, hmmer_e_value, n_threads_per_process, data):
    hmm_db = db_folder / 'amp_no_pfam.hmm'

    with tempfile.NamedTemporaryFile(suffix='.domtblout.txt', delete=False) as f:
        domtblout_path = Path(f.name).resolve()

    try:
        response = subprocess.run(
            [
                'hmmsearch',
                '--noali',
                '-o', '/dev/null',
                '--domtblout', domtblout_path.as_posix(),
                '--cpu', f'{n_threads_per_process}',
                '-E', f'{hmmer_e_value}',
                hmm_db.resolve().as_posix(),
                protein_path.resolve().as_posix(),
            ],
            capture_output=True,
        )
        if response.returncode != 0:
            stderr_txt = response.stderr.decode('utf-8')
            logger.error(f'Error while running `hmmsearch`: {stderr_txt}')
            return

        # Process hmmer output
        with domtblout_path.open() as f:
            parser = Hmmer3DomtabHmmqueryParser(f)
            for record in parser:
                for protein_id, hit in record.items:
                    hmm_query = hit.query_id
                    for hit_instance in hit:
                        evalue = hit_instance.evalue
                        bitscore = hit_instance.bitscore

                        data['assembly_accession'].append(assembly_accession)
                        data['protein_id'].append(protein_id)
                        data['amp_id'].append(hmm_query)
                        data['amp_desc'].append(None)
                        data['evalue'].append(evalue)
                        data['bitscore'].append(bitscore)

    finally:
        if domtblout_path.is_file():
            domtblout_path.unlink()


def screen_for_proteins(protein_path, db_folder, assembly_accession, n_threads_per_process, data):
    tmp_folder = Path(tempfile.gettempdir())
    protein_queries = db_folder / 'amp.fasta'

    with tempfile.NamedTemporaryFile(suffix='.results.txt', delete=False) as f:
        results_path = Path(f.name).resolve()
    
    try:
        try_number = 1
        num_tries = 3
        while try_number <= num_tries:
            response = subprocess.run(
                [
                    'mmseqs',
                    'easy-search',
                    protein_queries.resolve().as_posix(),
                    protein_path.resolve().as_posix(),
                    results_path.as_posix(),
                    tmp_folder.resolve().as_posix(),
                    '--threads', f'{n_threads_per_process}',
                    '--format-output', 'query,target,evalue,bits',
                ],
                capture_output=True,
            )
            if response.returncode != 0:
                if try_number == num_tries:
                    stderr_txt = response.stderr.decode('utf-8')
                    logger.error(f'Error while running `mmseqs easy-search`: {stderr_txt}')
                    return
                else:
                    try_number += 1
                    continue
            else:
                break

        mmseqs2_output_columns = ['query', 'target', 'evalue', 'bitscore']
        mmseqs2_output = pd.read_csv(results_path, sep='\t', header=None, names=mmseqs2_output_columns)

        for row in mmseqs2_output.itertuples(index=False):
            data['assembly_accession'].append(assembly_accession)
            data['protein_id'].append(row.target)
            data['amp_id'].append(row.query)
            data['amp_desc'].append(None)
            data['evalue'].append(row.evalue)
            data['bitscore'].append(row.bitscore)
    
    finally:
        if results_path.is_file():
            results_path.unlink()


def append_to_output_file(data, output_path, header):
    df = pd.concat(data, ignore_index=True)
    df.to_csv(
        output_path, 
        mode='a', 
        header=header, 
        index=False,
    )


def extract_accession_from_path_name(path_name : str):
    m = re.match(r'^(GC[AF]_[^_]+)_.+$', path_name)
    return m[1]


if __name__ == '__main__':
    main()
