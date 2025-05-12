"""
Compute pI and AA distribution for all proteins in input fasta file.
"""
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

from isoelectric import ipc
import numpy as np
import pandas as pd
from Bio import SeqIO, SeqRecord


logger = logging.getLogger()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_fasta', 
        help='Path to input fasta file', 
        type=Path,
        required=True,
    )
    parser.add_argument(
        '-o', '--output_csv', 
        help='Path to output CSV file', 
        type=Path,
        required=True,
    )
    parser.add_argument('--n_cpu', type=int, default=4)

    args = parser.parse_args()
    input_fasta = args.input_fasta
    output_csv = args.output_csv
    n_cpu = args.n_cpu

    if not input_fasta.is_file():
        logger.error(f'Input fasta does not exist: {input_fasta}')
        sys.exit(1)
    elif not output_csv.parent.is_dir():
        logger.error(f'Output folder does not exist: {output_csv.parent}')
        sys.exit(1)

    # Create as many temp file as we have CPUs
    temp_fasta_files = []
    temp_fasta_filepaths = []
    for _ in range(n_cpu):
        f = tempfile.NamedTemporaryFile('a', suffix='.fasta', encoding='utf-8', delete=False)
        filepath = Path(f.name).resolve()
        temp_fasta_files.append(f)
        temp_fasta_filepaths.append(filepath)

    try:
        # Split large fasta files into smaller temporary fasta files
        logger.info(f'Splitting large fasta file into {len(temp_fasta_files)} temporary fasta files')
        current_file_index = 0
        for record in SeqIO.parse(input_fasta, 'f'):
            SeqIO.write(record, temp_fasta_files[current_file_index], 'fasta')
            current_file_index = (current_file_index + 1) % len(temp_fasta_files)

        # Closing files
        for f in temp_fasta_files:
            f.close()

        logger.info('Processing each fasta file independently in a separate process')

        processes = []
        queue = Queue()
        for i in range(n_cpu):
            p = Process(target=worker_main, args=(
                i,
                temp_fasta_filepaths[i]
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
            with output_csv.open('w') as f:
                sorted_paths = [p.as_posix() for _, p in partial_paths]
                returncode = subprocess.call(
                    ['cat'] + sorted_paths, 
                    stdout=f,
                )
                if returncode != 0:
                    logger.error(f'Error while concatenating files')
                    sys.exit(1)
        finally:
            # Remove temp CSV files
            for _, p in partial_paths:
                if p.is_file():
                    p.unlink()

    finally:
        # Delete temp fasta files
        for p in temp_fasta_filepaths:
            if p.is_file():
                p.unlink()

    logger.info('DONE')
    sys.exit(0)


def worker_main(
    worker_ix : int, 
    fasta_path : Path,
    queue : Queue,
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    logger.info(f'Worker {worker_ix+1}: STARTING')

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        output_path = Path(f.name).resolve()

    header = worker_ix == 0

    data = []
    save_every = 1000
    for i, record in enumerate(SeqIO.parse(fasta_path, 'fasta')):
        if i == 0 or (i+1) % save_every == 0:
            logger.info(f'Worker {worker_ix + 1} | Processing record {i+1:,}')

        out_series = process_record(record)
        data.append(out_series)

        if len(data) >= save_every:
            append_to_output_file(data, output_path, header)
            header = False
            data = []

    if len(data) > 0:
        append_to_output_file(data, output_path, header)

    queue.put((worker_ix, output_path))
    logger.info(f'Worker {worker_ix+1}: DONE')


def append_to_output_file(data, output_path, header):
    df = pd.DataFrame(data)
    df.to_csv(
        output_path, 
        mode='a', 
        header=header, 
        index=False,
    )


def process_record(record : SeqRecord) -> pd.Series:
    AAs = 'ACDEFGHIKLMNPQRSTVWY'
    sequence = str(record.seq).upper()
    if sequence[-1] == '*':
        sequence = sequence[:-1]

    molecular_weight = np.round(ipc.calculate_molecular_weight(sequence), 2)
    pI = np.round(ipc.predict_isoelectric_point(sequence), 2)

    data = {
        'id': record.id,
        'protein_length': len(sequence),
        'molecular_weight': molecular_weight,
        'pI': pI,
    }
    for aa in AAs:
        data[aa] = sequence.count(aa)

    return pd.Series(data)


if __name__ == '__main__':
    main()
