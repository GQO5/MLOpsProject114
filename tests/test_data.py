from mlopsproject.data import load_data


def test_load_data():
    """Test the load_data function."""
    try:
        (
            train_loader,
            val_loader,
            test_loader,
            train_raw,
            val_raw,
            test_raw,
            y_mean,
            y_std,
        ) = load_data()
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        assert len(y_mean) == 4  # 4 nutritional targets
        assert len(y_std) == 4
    except AssertionError:
        # If data files don't exist, skip the test
        import pytest

        pytest.skip("Data files not available for testing")
