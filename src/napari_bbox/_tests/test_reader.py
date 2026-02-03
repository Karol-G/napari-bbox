import numpy as np

from .._reader import napari_get_reader


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    my_test_file = str(tmp_path / "myfile.csv")
    # 2 bboxes in 2D => shape (N, 2*D) == (2, 4)
    original_data = np.array(
        [
            [1, 2, 10, 20],
            [3, 4, 30, 40],
        ],
        dtype=float,
    )
    np.savetxt(my_test_file, original_data, delimiter=",")

    # try to read it back in
    reader = napari_get_reader(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # make sure it's the same as it started (reshaped to N x 2 x D)
    expected = original_data.reshape(2, 2, 2)
    np.testing.assert_allclose(expected, layer_data_tuple[0])


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
