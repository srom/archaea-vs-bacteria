"""
Script to identify whether a protein is on a plasmid or on the chromosome. 

For complete genomes (ncbi_assembly_level == "Complete Genome"), identification is straightforward. 
For non-complete genomes, simply report the name and length of the contig / scaffold that the protein is part of.

Input: CSV file containing at least these two columns: assembly_accession, protein_id
Output: CSV file with columns: 
- assembly_accession
- protein_id
- contig_id
- contig_length
- location (one of Chromosome or Plasmid or N/A)
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from multiprocessing import Process, Queue
from queue import Empty
import gzip
import re
import tempfile
from typing import List

import gffutils
import numpy as np
import pandas as pd
from Bio import SeqIO


logger = logging.getLogger()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_path',
        help='Path to CSV file with at least two columns: assembly_accession, protein_id', 
        type=Path,
        required=True,
    )
    parser.add_argument(
        '-a', '--assemblies', 
        help='Path to base folder containing assemblies', 
        type=Path,
        required=True,
    )
    parser.add_argument(
        '-o', '--output_path', 
        help='Path to output file', 
        type=Path,
        required=True,
    )
    parser.add_argument('--n_cpus', type=int, default=4)

    args = parser.parse_args()
    base_folder = args.assemblies
    input_path = args.input_path
    output_path = args.output_path
    n_cpus = args.n_cpus

    if not base_folder.is_dir():
        logger.error(f'Assemblies folder does not exist: {base_folder}')
        sys.exit(1)
    elif not input_path.is_file():
        logger.error(f'Input CSV file does not exist: {input_path}')
        sys.exit(1)
    elif not output_path.parent.is_dir():
        logger.error(f'Output folder does not exist: {output_path.parent}')
        sys.exit(1)

    input_df = pd.read_csv(input_path)

    if 'assembly_accession' not in input_df.columns:
        logger.error(f'Column "assembly_accession" not in input CSV')
        sys.exit(1)
    elif 'protein_id' not in input_df.columns:
        logger.error(f'Column "protein_id" not in input CSV')
        sys.exit(1)

    accessions = set(input_df['assembly_accession'].unique())
    paths = sorted([
        p for p in base_folder.iterdir()
        if (
            p.is_dir() and 
            p.name.startswith('GC') and 
            '_'.join(p.name.split('_')[:2]) in accessions
        )
    ])

    logger.info(f'Number of proteins to process: {len(input_df):,}')
    logger.info(f'Number of assemblies to process: {len(paths):,}')

    if len(paths) == 0:
        sys.exit(0)

    n_processes = min(n_cpus, len(paths))
    n_per_process = int(np.ceil(len(paths) / n_processes))

    processes = []
    queue = Queue()
    for i in range(n_processes):
        start = i * n_per_process
        end = start + n_per_process
 
        p = Process(target=worker_main, args=(
            i,
            input_path,
            paths[start:end],
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

    logger.info('DONE')
    sys.exit(0)


def worker_main(
    worker_ix : int, 
    input_path : os.PathLike, 
    paths : List[os.PathLike], 
    queue : Queue,
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    logger.info(f'Worker {worker_ix+1}: STARTING')

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        output_path = Path(f.name).resolve()

    header = worker_ix == 0

    data = []
    save_every = 100
    for i, path in enumerate(paths):
        if i == 0 or (i+1) % save_every == 0 or (i+1) == len(paths):
            logger.info(f'Worker {worker_ix + 1} | Processing assembly {i+1:,} / {len(paths):,}')

        out_df = plasmid_vs_chromosome(input_path, path)

        if out_df is not None and len(out_df) > 0:
            data.append(out_df)

        if len(data) >= save_every:
            append_to_output_file(data, output_path, header)
            header = False
            data = []

    if len(data) > 0:
        append_to_output_file(data, output_path, header)

    queue.put((worker_ix, output_path))
    logger.info(f'Worker {worker_ix+1}: DONE')


def plasmid_vs_chromosome(input_path, assembly_folder):
    assembly_accession = extract_accession_from_path_name(assembly_folder.name)
    is_refseq = assembly_accession.startswith('GCF_')

    assembly_report_path = assembly_folder / f'{assembly_folder.name}_assembly_report.txt'
    genome_path = assembly_folder / f'{assembly_folder.name}_genomic.fna.gz'
    gff_path_gz = assembly_folder / f'{assembly_folder.name}_genomic.gff.gz'

    if not assembly_report_path.is_file():
        logger.error(f'No assembly report for assembly {assembly_accession}')
        return None
    elif not genome_path.is_file():
        logger.error(f'No genome file for assembly {assembly_accession}')
        return None
    elif not gff_path_gz.is_file():
        logger.error(f'No GFF for assembly {assembly_accession}')
        return None

    protein_ids = set(pd.read_csv(input_path).sort_values(
        ['assembly_accession', 'protein_id']
    ).set_index('assembly_accession').loc[
        [assembly_accession]
    ]['protein_id'].unique())

    assembly_report = pd.read_csv(
        assembly_report_path, 
        sep='\t',
        comment='#',
        header=None,
        names=[
            'Sequence-Name',
            'Sequence-Role',
            'Assigned-Molecule', 
            'Assigned-Molecule-Location/Type',
            'GenBank-Accn',
            'Relationship',
            'RefSeq-Accn',
            'Assembly-Unit',
            'Sequence-Length',
            'UCSC-style-name',
        ]
    ).set_index('RefSeq-Accn' if is_refseq else 'GenBank-Accn', drop=True)

    contig_id_to_length = {}
    with gzip.open(genome_path.as_posix(), 'rt') as f:
        for record in SeqIO.parse(f, 'fasta'):
            contig_id_to_length[record.id] = len(record.seq)

    median_contig_length = np.median(list(contig_id_to_length.values()))

    output_data = {
        'assembly_accession': [],
        'protein_id': [],
        'contig_id': [],
        'contig_length': [],
        'median_contig_length': [],
        'location': [],
    }

    with tempfile.NamedTemporaryFile(suffix='.gff', delete=False) as gff_file:
        gff_path = Path(gff_file.name).resolve()
        returncode = subprocess.call(
            ['gzip', '-cd', gff_path_gz.resolve().as_posix()],
            stdout=gff_file,
        )

        try:
            if returncode != 0:
                logger.error(f'Error while decompressing {gff_path_gz}')
                return None
            
            gff_db = gffutils.create_db(
                gff_path.as_posix(), 
                dbfn=':memory:', 
                force=True, 
                keep_order=True, 
                merge_strategy='merge', 
                sort_attribute_values=True,
            )
            
            query = "select * from features where featuretype = 'CDS'"
            features = [
                gffutils.Feature(**row)
                for row in gff_db.execute(query)
            ]
            seen_protein_ids = set()
            for f in features:
                contig_id = f.seqid
                protein_id = None
                if 'protein_id' in f.attributes:
                    if 'protein_id' in f.attributes:
                        protein_id = f.attributes['protein_id'][0]
                    elif 'Name' in f.attributes:
                        protein_id = f.attributes['Name'][0]
                    elif 'ID' in f.attributes:
                        if is_refseq:
                            protein_id = f.attributes['ID'][0].replace('cds-', '')
                        else:
                            protein_id = f.attributes['ID'][0]
                            if protein_id.startswith('cds-'):
                                n = protein_id.split('_')[-1]
                                protein_id = f'{contig_id}_{n}'

                if protein_id in protein_ids and protein_id not in seen_protein_ids:
                    try:
                        location = assembly_report.loc[contig_id, 'Assigned-Molecule-Location/Type']
                    except KeyError:
                        logger.error(f'Contig {contig_id} not found for {protein_id}@{assembly_accession}')
                        location = None

                    if location == 'na':
                        location = None

                    contig_length = contig_id_to_length.get(contig_id)
                    
                    output_data['assembly_accession'].append(assembly_accession)
                    output_data['protein_id'].append(protein_id)
                    output_data['contig_id'].append(contig_id)
                    output_data['contig_length'].append(contig_length)
                    output_data['median_contig_length'].append(median_contig_length)
                    output_data['location'].append(location)

                    seen_protein_ids.add(protein_id)
            
            missing_protein_ids = sorted(set(protein_ids) - seen_protein_ids)

            for missing_protein_id in missing_protein_ids:
                output_data['assembly_accession'].append(assembly_accession)
                output_data['protein_id'].append(missing_protein_id)
                output_data['contig_id'].append(None)
                output_data['contig_length'].append(None)
                output_data['median_contig_length'].append(median_contig_length)
                output_data['location'].append(None)

        finally:
            if gff_path.is_file():
                gff_path.unlink()

    return pd.DataFrame.from_dict(output_data).sort_values(['assembly_accession', 'protein_id'])


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
