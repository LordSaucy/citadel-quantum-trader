# setup.py
from setuptools import setup, find_packages

setup(
    name="citadel-quantum-trader",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "cqt-ingest = src.data_ingest.collector:run",
        ],
    },
    # … other metadata …
)
