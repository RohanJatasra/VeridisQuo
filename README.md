# Veridis Quo
**Where is the truth?** At a time when video is gradually ceasing to be viable evidence, with the constant progress of tools for manipulating this media, we feel the need to seek new ways of discerning reality from fake. 
Veridis Quo is a neural network designed to **detect deepfake videos** and **highlight the altered areas** in each frame.

TODO...

## Param
```bash
Total params: 40,798,098
Trainable params: 40,798,098
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 300.57
Params size (MB): 155.63
Estimated Total Size (MB): 456.78
```

## Prerequisites

Install [uv](https://docs.astral.sh/uv/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# OR
pip install uv
```

## Launch

```bash
./scripts/dev.sh
# OR
uv sync
uv run python src/main.py
```

## Tests

```bash
./scripts/run_tests.sh
# OR
uv sync --extra dev
uv run pytest tests/ -v
```





## Training

Concernant le training du modele
Nous avons entrainé le modele sur un RTX 3090 (avec cuda) pendatn environ 4h. Nous avons utilisé le provider de gpu [vast.ai](https://vast.ai/)

Concernant les données . Nous somme partie du dataset existant trouvé sur kaggle [FaceForensics++ Dataset (C23)](https://www.kaggle.com/datasets/xdxd003/ff-c23) contenant en tout 7000 videos avec des nombreuses techniques de deepfake comme : Face2Face , FaceShifter , FaceSwap , NeuralTextures.
Nous avons extrait les frames et les visages de ces videos pour creer un dataset disponible sur : [VeridisQuo Dataset](https://www.kaggle.com/datasets/gazeux330000/deepfake-dataset/settings)

| type  | nb_sample | split |
|-------|-----------|-------|
| train | 499965    | 0.7   |
| test  | 107620    | 0.15  |
| eval  | 108853    | 0.15  |

Concernant le fichier d'entrainement , il se situe dans le module ```training/trainer.py```

Nous avons utilisé la config suivante : 

| Paramètre                | Valeur                   |
|--------------------------|--------------------------|
| batch_size               | 64                       |
| learning_rate            | 0.0001                   |
| min_learning_rate        | 0.000001                 |
| num_epochs               | 7                        |
| optimizer                | AdamW                    |
| scheduler                | Warmup + Cosine Annealing|
| use_automixed_precision  | false                    |
| warmup_epochs            | 2                        |
| weight_decay             | 0.0001                   |
| loss_func                | *(à préciser)*           |


Le modele est maintenant disponible sur huggingFace: [VeridisQuo Model](https://huggingface.co/Gazeux33/VeridisQuo/tree/main)

![image](./assets/training/train_acc.png)
![image](./assets/training/train_loss.png)
![image](./assets/training/val_acc.png)
![image](./assets/training/val_loss.png)
![image](./assets/training/train_lr.png)







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