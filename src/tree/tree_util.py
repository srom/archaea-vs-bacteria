"""
Tree utilities.
"""
import re
import copy
from typing import Type, Dict, Set

from Bio import Phylo


RS_PATTERN = r'^RS_(GCF_[0-9\.]+)$'
GB_PATTERN = r'^GB_(GCA_[0-9\.]+)$'


def rename_gtdb_leaves_from_refseq_to_genbank_ids(
    tree: Phylo.BaseTree.Tree, 
    rs_to_gb: Dict[str, str],
) -> Phylo.BaseTree.Tree:
    """
    GTDB [1] trees use IDs of the following form:
      - RS_GCF_* for RefSeq ids
      - GB_GCA_* for GenBank ids

    This function takes in a GTDB tree and convert all IDs to 
    their GenBank format without the GB_ prefix.

    RefSeq ids are converted to GenBank using the mapping provided.
    If a RefSeq id is missing from the map, the leaf is left unchanged.
    Returns a copy of the original tree.

    [1] https://gtdb.ecogenomic.org/ 
    """
    out_tree = copy.deepcopy(tree)

    for leaf in out_tree.get_terminals():
        name = leaf.name

        gb_match = re.match(GB_PATTERN, name)
        if gb_match is not None:
            leaf.name = gb_match[1]
            continue

        rs_match = re.match(RS_PATTERN, name)
        if rs_match is not None:
            rs_id = rs_match[1]

            if rs_id in rs_to_gb:
                gb_id = rs_to_gb[rs_id]
                leaf.name = gb_id

    return out_tree


def prune_leaves_with_unknown_id(
    tree: Phylo.BaseTree.Tree, 
    id_set: Set[str],
) -> Phylo.BaseTree.Tree:
    """
    Prune tree leaves that do not match any of the known ids.
    Returns a copy of the original tree.
    """
    out_tree = copy.deepcopy(tree)

    leaves = tree.get_terminals()
    for leaf in leaves:
        if leaf.name not in id_set:
            try:
                out_tree.prune(leaf.name)
            except ValueError:
                print(type(leaf), leaf.name)
                raise

    return out_tree


def make_phylogenetic_level_tree(tree, phylogenetic_prefix='c__', keep_prefix=False):
    class_nodes = []
    for node in tree.get_nonterminals():
        if node.name is not None and phylogenetic_prefix in node.name:
            class_nodes.append(node.name)

    out_tree = copy.deepcopy(tree)
    for c in class_nodes:
        node = out_tree.find_any(name=c)
        
        if not node.is_terminal():
            while True:
                if node.is_terminal():
                    break

                for n in node.get_terminals():
                    if n.is_terminal():
                        out_tree.collapse(n)

        node.name = re.match(f'^.+({phylogenetic_prefix}[^;]*).*$', c)[1].strip()

    for leaf in out_tree.get_terminals():
        if not leaf.name.startswith(phylogenetic_prefix):
            out_tree.prune(leaf)
        elif not keep_prefix:
            leaf.name = leaf.name.replace(phylogenetic_prefix, '')

    return out_tree
