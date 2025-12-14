# Veridis Quo
**Where is the truth?** At a time when video is gradually ceasing to be viable evidence, with the constant progress of tools for manipulating this media, we feel the need to seek new ways of discerning reality from fake. 
Veridis Quo is a neural network designed to **detect deepfake videos** and **highlight the altered areas** in each frame.

TODO...

## Launch

```bash
./scripts/dev.sh
#OR
cd veridisquo/src
poetry install
poetry run python main.py
```

## Tests

```bash
./scripts/run_tests.sh
```

## Project Structure
```
veridisquo/
├── scripts/                   # Helper scripts
│   ├── dev.sh                 # Development environment setup script
│   └── run_tests.sh           # Unit tests execution script
├── src/                       # Source code
│   ├── preprocessing/         # Data preprocessing module
│   ├── spatial/               # Feature extraction module
│   ├── data/                  # Dataset loaders module
│   ├── frequal/               # Frequency analysis module
│   └── main.py                # Main script
├── tests/                     # Unit tests
│   ├── preprocessing/         # Tests for data preprocessing
│   └── spatial/               # Tests for feature extraction
├── README.md                  # Project documentation
└── pyproject.toml             # Project configuration
```