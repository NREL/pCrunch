
from pCrunch import AeroelasticOutput


def test_array_input(array_input, magnitude_channels):

    channels, data = array_input
    output = AeroelasticOutput(
        data, channels, dlc="Test Data", magnitude_channels=magnitude_channels
    )

    assert output.data.shape
    assert output.channels.shape
    assert "Wind" in output.channels


def test_dict_input(dict_input, magnitude_channels):

    output = AeroelasticOutput(
        dict_input, dlc="Test Data", magnitude_channels=magnitude_channels
    )

    assert output.data.shape
    assert output.channels.shape
    assert "Wind" in output.channels
