# Tested environment

Verified with CPython 3.11 on macOS using the AlphaResearch Conda environment.
These are observed direct dependency versions, not a cross-platform lockfile.
`pyproject.toml` remains the installation source; GPU execution is optional.

```text
numpy==2.4.6
pandas==2.3.3
pyqlib==0.9.7
pyyaml==6.0.3
requests==2.34.2
openai==3.7.0
jsonschema==4.26.0
tqdm==4.70.0
torch==2.14.0
gpytorch==1.15.2
matplotlib==3.11.1
pytest==9.1.1
zss==1.2.0
gunicorn==23.0.0
```

The test suite and saved-data plotting need no API credentials or market data.
Real evaluation requires the external Qlib provider and FFO service.
