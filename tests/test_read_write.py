""" Test faithfulness of read/write
"""

import unittest
import tempfile

from pandas import pd

from hypothesis import given, strategies

from sleep_analysis.data import read_data, write_data


@strategies.composite
def fly_dataframes(draw):
    """ Composite strategy for producing fly dataframes, as might be returned by read
    """
    n_rows = draw(strategies.integers(min_value=1))
    n_fly_cols = draw(strategies.integers(min_value=1, max_value=512))

    data = {}

    data['light'] = draw(
        strategies.lists(
            strategies.integers(min_value=0, max_value=1),
            min_size=n_rows,
            max_size=n_rows
        )
    )

    for idx in range(n_fly_cols):
        data[f'fly_{idx}'] = draw(
            strategies.lists(
                strategies.integers(min_value=0),
                min_size=n_rows,
                max_size=n_rows
            )
        )

    fly_df = pd.DataFrame.from_dict(data)

    return fly_df


class TestReadWrite(unittest.TestCase):

    @given(fly_dataframes())
    def test_read_write_faithfulness(self, fly_df):
        """ Test io faithfulness
        """
        # implicit assert directory already exists
        with tempfile.NamedTemporaryFile() as tmp_file:
            write_data(tmp_file.name, fly_df)
            reconstructed_df = read_data(tmp_file.name)

        self.assertEqual(fly_df, reconstructed_df)
