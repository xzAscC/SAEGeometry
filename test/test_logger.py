from src import setup_logger


def test_get_logger():
    logger = setup_logger()
    assert logger is not None
    logger.info("Test logger")
