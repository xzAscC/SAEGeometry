from src import get_logger

def test_get_logger():
    logger = get_logger()
    assert logger is not None