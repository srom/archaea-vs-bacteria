import unittest

from Bio import Phylo

from .tree_util import (
    rename_gtdb_leaves_from_refseq_to_genbank_ids,
    prune_leaves_with_unknown_id,
)


class TestTreeUtil(unittest.TestCase):

    def setUp(self):
        self.tree = Phylo.read('fixtures/bac_fixture.tree', 'newick')

    def test_rename_gtdb_leaves_from_refseq_to_genbank_ids(self):
        tree = rename_gtdb_leaves_from_refseq_to_genbank_ids(
            self.tree,
            {
                'GCF_001020875.1': 'GCA_000230735.3',
            },
        )

        leaves = tuple(l.name for l in tree.get_terminals())

        self.assertTupleEqual(
            (
                'GCA_000230735.3',
                'RS_GCF_902498005.1',
                'RS_GCF_014472415.1',
                'RS_GCF_002813455.1',
                'GCA_001919195.1',
            ),
            leaves,
        )

    def test_prune_leaves_with_unknown_id(self):
        tree = prune_leaves_with_unknown_id(
            self.tree,
            {'RS_GCF_902498005.1', 'RS_GCF_002813455.1'},
        )

        leaves = tuple(l.name for l in tree.get_terminals())

        self.assertTupleEqual(
            (
                'RS_GCF_902498005.1',
                'RS_GCF_002813455.1',
            ),
            leaves,
        )


if __name__ == '__main__':
    unittest.main()
