"""
Script to run [SignalP 6.0](https://github.com/fteufel/signalp-6.0) to predict signal peptides.

SignalP 6.0 must be installed: [installation instructions](https://github.com/fteufel/signalp-6.0/blob/main/installation_instructions.md). 
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from signalp.predict import predict, MODEL_DIR
from signalp.utils import (
    get_cleavage_sites, 
    postprocess_probabilities, 
    postprocess_viterbi, 
    get_cleavage_site_probs, 
    resolve_viterbi_marginal_conflicts,
    tokenize_sequence,
)
from Bio import SeqIO

logger = logging.getLogger()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--fasta', 
        help='Path to input fasta file', 
        type=Path,
        required=True,
    )
    parser.add_argument(
        '-o', '--output_path', 
        help='Path to output file', 
        type=Path,
        required=True,
    )
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--n_processes', type=int, default=4)
    args = parser.parse_args()

    input_fasta = args.fasta
    output_path = args.output_path
    batch_size = args.batch_size
    n_processes = args.n_processes
    use_gpu = args.use_gpu

    if use_gpu and not torch.cuda.is_available():
        logger.error('Option --use_gpu was used but no GPU available - aborting')
        sys.exit(1)
    elif use_gpu:
        logger.info(f'GPU available: {torch.cuda.is_available()}')
        logger.info(f'torch.cuda.device_count() = {torch.cuda.device_count()}')

    torch.set_num_threads(n_processes)

    if not input_fasta.is_file():
        logger.error(f'Input fasta file does not exist: {args.fasta}')
        sys.exit(1)
    elif not output_path.parents[0].is_dir():
        logger.error(f'Output folder does not exist: {output_path.parents[0].as_posix()}')
        sys.exit(1)

    model_path = Path(MODEL_DIR) / 'distilled_model_signalp6.pt'
    signalp_model = torch.jit.load(model_path, map_location=torch.device('cuda'))

    fasta_iter = SeqIO.parse(input_fasta, 'fasta')

    protein_records = []
    include_header = True
    for i, protein_record in enumerate(fasta_iter):
        protein_records.append(protein_record)

        if len(protein_records) == batch_size:
            n_records = run_signalp(signalp_model, protein_records, output_path, batch_size, include_header)
            protein_records = []
            if n_records > 0:
                include_header = False

            logger.info(f'Processed proteins: {i+1:,}')

    if len(protein_records) > 0:
        run_signalp(signalp_model, protein_records, output_path, include_header)
        logger.info(f'Processed proteins: {i+1:,}')

    logger.info('DONE')
    sys.exit(0)


def run_signalp(signalp_model, protein_records, output_path, batch_size, include_header):
    """
    Adapted from main script of SignalP.
    """
    domain_id = 'other'  # prokaryotes

    # Read protein file and make torch tensors
    identifiers, _, input_ids, input_mask = model_inputs_from_record(protein_records, domain_id)

    # Predict signal peptide
    global_probs, marginal_probs, viterbi_paths = predict(signalp_model, input_ids, input_mask, batch_size=batch_size)

    # Infer the cleavage sites from the viterbi paths.
    cleavage_sites =  get_cleavage_sites(viterbi_paths) #this uses labels before post-processing.

    # Simplify the marginal probabilities and merge classes into Sec/SPI for eukarya.
    marginal_probs = postprocess_probabilities(marginal_probs, domain_id)

    # Same simplifications for viterbi paths.
    viterbi_paths = postprocess_viterbi(viterbi_paths, domain_id)

    # Resolve edge case discrepancies between viterbi decoding and marginal probabilities.
    resolve_viterbi_marginal_conflicts(global_probs, marginal_probs, cleavage_sites, viterbi_paths)

    # Retrieve the probability of each predicted cleavage site (=probability of preceding label)
    cs_probs = get_cleavage_site_probs(cleavage_sites, viterbi_paths, marginal_probs)

    # Write results and return the number of rows written
    return append_to_output_file(
        identifiers, 
        global_probs, 
        cleavage_sites, 
        cs_probs, 
        output_path,
        include_header,
    )


def model_inputs_from_record(fasta_records: str, domain_id: str):
    """
    Adapted from SignalP codebase:
    Parse a fasta file to input id + mask tensors.
    Pad all seqs to full length (73: 70+special tokens).
    traced model requires that, it was traced with 73 as fixed length
    """
    identifiers = [r.id for r in fasta_records]
    sequences = [str(r.seq) for r in fasta_records]

    # Truncate
    input_ids = [x[:70] for x in sequences]
    input_ids =  [tokenize_sequence(x, domain_id) for x in input_ids]
    input_ids = [x + [0] * (73-len(x)) for x in input_ids]

    # Mask
    input_ids = np.vstack(input_ids)
    input_mask = (input_ids > 0) * 1

    return (
        identifiers, 
        sequences, 
        torch.LongTensor(input_ids), 
        torch.LongTensor(input_mask),
    )


def append_to_output_file(
    identifiers, 
    global_probs, 
    cleavage_sites, 
    cs_probs, 
    output_path,
    include_header,
):
    data = {
        'protein_id': [],
        'signal_peptide': [],
        'probability': [],
        'cleavage_site': [],
        'cleavage_site_prob': [],
    }
    label_dict = {
        1: 'Sec/SPI', 
        2: 'Sec/SPII', 
        3: 'Tat/SPI', 
        4: 'Tat/SPII',
        5: 'Sec/SPIII',
    }

    pred_label_ids = np.argmax(global_probs, axis=1)

    for idx, identifier in enumerate(identifiers):
        label_id = pred_label_ids[idx]
        if label_id == 0:  # no signal peptide
            continue

        prediction = label_dict[label_id]
        probability = global_probs[idx][label_id]

        cs = cleavage_sites[idx] if cleavage_sites[idx] != -1 else None
        cs_prob = cs_probs[idx]
        cs_str = f'{cs}-{cs+1}' if cs is not None else None
        cs_prob_str = f'{cs_prob:.6f}' if cs is not None else None

        data['protein_id'].append(identifier)
        data['signal_peptide'].append(prediction)
        data['probability'].append(f'{probability:.6f}')
        data['cleavage_site'].append(cs_str)
        data['cleavage_site_prob'].append(cs_prob_str)

    n_records = len(data['protein_id'])
    if n_records > 0:
        pd.DataFrame.from_dict(data).to_csv(
            output_path,
            index=False,
            mode='a',
            header=include_header,
        )

    return n_records


if __name__ == '__main__':
    main()
