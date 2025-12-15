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
│   ├── spatial/               # Spatial feature extraction module
│   ├── data/                  # Dataset loaders module
│   ├── frequal/               # Frequency feature extraction module
│   └── main.py                # Main script
├── tests/                     # Unit tests
│   ├── preprocessing/         # Tests for data preprocessing
│   ├── spatial/               # Tests for spatial feature extraction
│   └── frequal/               # Tests for frequency feature extraction
├── README.md                  # Project documentation
└── pyproject.toml             # Project configuration
```

### Frequency Analysis Module Structure
```
  BaseFrequencyExtractor (ABC)
  │   ├── extract() [template method]
  │   ├── _preprocess_image() [abstract]
  │   ├── _apply_transform() [abstract]
  │   └── _extract_features() [abstract]
  ├── BaseFFTExtractor(BaseFrequencyExtractor) (ABC)
  │   │   ├── Constantes FFT
  │   │   ├── _apply_transform() → call _apply_fft()
  │   │   ├── _extract_features() → call _extract_frequency_features()
  │   │   ├── _apply_fft() [abstract]
  │   │   └── _extract_frequency_features() [abstract]
  │   ├── NpFFTExtractor(BaseFFTExtractor)
  │   │   └── NumPy implementations
  │   └── TorchFFTExtractor(nn.Module, BaseFFTExtractor)
  │       └── PyTorch implementations
  └── BaseDCTExtractor(BaseFrequencyExtractor) (ABC)
      │   ├── Constantes DCT (AVAILABLE_AGGREGATION_METHODS, DEFAULT_BLOCK_SIZE, etc.)
      │   ├── _apply_transform() → call _extract_dct_blocks()
      │   ├── _extract_features() → call _aggregate_block_features()
      │   ├── _extract_dct_blocks() [abstract]
      │   └── _aggregate_block_features() [abstract]
      ├── NpDCTExtractor(BaseDCTExtractor)
      │   └── NumPy implementations + 3 aggregation methods
      └── TorchDCTExtractor(BaseDCTExtractor)
          └── PyTorch implementations
```