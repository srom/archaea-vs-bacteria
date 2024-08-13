import tempfile
import unittest

from .itol_annotation import itol_binary_annotations


class TestITolAnnotation(unittest.TestCase):

    def setUp(self):
        with open('fixtures/itol_binary.txt', 'r') as f:
            self.binary_target_str = f.read()

    def test_itol_binary_annotations(self):
        with tempfile.NamedTemporaryFile() as f:
            itol_binary_annotations(
                data=[
                    ['GCA_001294575.1', '1'],
                    ['GCA_000761155.1', '-1'],
                ],
                output_path=f.name,
                field_shapes=[2],
                field_labels=['Halocin C8 (TIGR04449)'],
                legend_title='Hits',
            )

            with open(f.name, 'r') as f2:
                res = f2.read()

        self.assertEqual(res, self.binary_target_str)


if __name__ == '__main__':
    unittest.main()
