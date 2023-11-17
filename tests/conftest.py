import pytest



def pytest_addoption(parser):
    parser.addoption(
        "--light", action="store_true", default=False, help="Skip tests using I/O heavy resources"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "ioheavy: mark test as requiring heavy I/O (/nas or otherwise)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--light"):
        # --light given in cli: skip slow tests
        skip_nas = pytest.mark.skip(reason="--light option was passed, skipping io-heavy tests")
        for item in items:
            if "ioheavy" in item.keywords:
                item.add_marker(skip_nas)

