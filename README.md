<table>
<tr>
<td width="60%" valign="middle">

# Veridis Quo
### *Where is the truth?*

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-VeridisQuo-yellow?style=for-the-badge)](https://huggingface.co/Gazeux33/VeridisQuo/tree/main)

**What is Veridis Quo?**
> A state-of-the-art neural network designed to **detect deepfake videos** and **highlight the altered areas** in each frame using explainable AI.

**The Era of Video Manipulation**
> At a time when video is gradually ceasing to be viable evidence, with the constant progress of tools for manipulating this media, we feel the need to seek new ways of discerning reality from fake.


</td>
<td width="40%" valign="middle" align="center">

<img src="./assets/logo/VeridisQuo.png" alt="VeridisQuo Logo" width="100%">

</td>
</tr>
</table>


# Pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1a1a2e', 'primaryTextColor': '#fff', 'lineColor': '#ffffff', 'background': '#0d1117', 'mainBkg': '#0d1117'}}}%%

flowchart TB
    subgraph Preprocess[" Preprocess "]
        A[MP4 Video] --> B[Frames extraction]
        B --> C[Faces detection]
        C --> D[Faces extraction]
    end

    D --> E
    D --> G
    D --> H

    subgraph Processing[" "]
        direction LR
        subgraph Spatial_Module[" Spatial Module "]
            E[Features ExtractionEfficientNet] -->|"1792, 7, 7"| F((Pooled))
        end

        subgraph Frequency_Module[" Frequency Module "]
            G[DCT Extractor8×8, freq_bands] -->|"512"| I((Concat))
            H[FFT Extractorradial=8, hann] -->|"512"| I
            I -->|"1024"| J[FusionMLP1024→512→1024]
        end
    end

    F -->|"1792"| K
    J -->|"1024"| K

    subgraph Classifier_Module[" Classifier Module "]
        K((Concat)) -->|"2816"| L[MLP]
        L --> M[Frames Aggregation]
        M --> N[Final Score]
    end

    E --> O
    D --> Q

    subgraph Gradcam_Module[" Gradcam Module "]
        O[Gradcam Computation] --> P[Gradcam Visualization]
        P --> Q((Remap))
        Q --> R[Final Manipulated Video]
    end

    %% Styling
    classDef preprocess fill:#1a1a2e,stroke:#f59e0b,color:#f59e0b
    classDef spatial fill:#1a1a2e,stroke:#10b981,color:#10b981
    classDef frequency fill:#1a1a2e,stroke:#3b82f6,color:#3b82f6
    classDef classifier fill:#1a1a2e,stroke:#ec4899,color:#ec4899
    classDef gradcam fill:#1a1a2e,stroke:#a855f7,color:#a855f7
    classDef default fill:#1a1a2e,stroke:#6b7280,color:#fff

    class B,C,D preprocess
    class E,F spatial
    class G,H,I,J frequency
    class K,L,M,N classifier
    class O,P,Q,R gradcam

    style Preprocess fill:#0d1117,stroke:#f59e0b,color:#f59e0b
    style Spatial_Module fill:#0d1117,stroke:#10b981,color:#10b981
    style Frequency_Module fill:#0d1117,stroke:#3b82f6,color:#3b82f6
    style Classifier_Module fill:#0d1117,stroke:#ec4899,color:#ec4899
    style Gradcam_Module fill:#0d1117,stroke:#a855f7,color:#a855f7
    style Processing fill:transparent,stroke:none
```

---

## Model Specifications

<table>
<tr>
<td width="33%" align="center">

![Parameters](https://img.shields.io/badge/Parameters-40.8M-blue?style=for-the-badge)

**Total Parameters**

All trainable

</td>
<td width="33%" align="center">

![Input](https://img.shields.io/badge/Input-224×224_RGB-green?style=for-the-badge)

**Input Resolution**

RGB images

</td>
<td width="34%" align="center">

![Features](https://img.shields.io/badge/Features-2816--dim-purple?style=for-the-badge)

**Feature Vector**

Combined spatial + frequency

</td>
</tr>
</table>

---




## Training
<table>
<tr>
<td width="60%" valign="top">

### Infrastructure
We trained the model on an **RTX 3090** (with CUDA) for approximately **4 hours**.
We used the GPU provider <a href="https://vast.ai/" target="_blank">vast.ai</a>

The training file is located in the `training/trainer.py` module

</td>
<td width="40%" valign="middle">

| Specs | |
|:---------|:---|
| **GPU** | ![GPU](https://img.shields.io/badge/RTX_3090-76B900?style=flat-square&logo=nvidia&logoColor=white) |
| **Framework** | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white) |
| **Duration** | ![Training](https://img.shields.io/badge/~4h-blue?style=flat-square&logo=clockify&logoColor=white) |

</td>
</tr>
</table>

---

## Dataset
<table>
<tr>
<td width="50%" valign="top">

### Source
We started from an existing dataset found on Kaggle:  
**<a href="https://www.kaggle.com/datasets/xdxd003/ff-c23" target="_blank">FaceForensics++ Dataset (C23)</a>**

Containing **7000 videos** with numerous deepfake techniques:
<p>
  <img src="https://img.shields.io/badge/Face2Face-6366f1?style=flat-square" alt="Face2Face"/>
  <img src="https://img.shields.io/badge/FaceShifter-22d3ee?style=flat-square" alt="FaceShifter"/>
  <img src="https://img.shields.io/badge/FaceSwap-a855f7?style=flat-square" alt="FaceSwap"/>
  <img src="https://img.shields.io/badge/NeuralTextures-f43f5e?style=flat-square" alt="NeuralTextures"/>
</p>

We extracted the frames and faces from these videos to create our dataset:  
**<a href="https://www.kaggle.com/datasets/gazeux330000/deepfake-dataset/settings" target="_blank">VeridisQuo Preprocessed Dataset</a>**

</td>
<td width="50%" valign="top">

### Distribution
| Split | Samples | Ratio |
|:-----:|--------:|:-----:|
| **Train** | 499,965 | 70% |
| **Test** | 107,620 | 15% |
| **Eval** | 108,853 | 15% |

<p><strong>Total: 716,438 images</strong></p>

</td>
</tr>
</table>

---

## Configuration & Results
<table>
<tr>
<td width="30%" valign="top">

### Configuration
| Parameter | Value |
|:----------|:------:|
| `batch_size` | 64 |
| `learning_rate` | 0.0001 |
| `min_learning_rate` | 0.000001 |
| `num_epochs` | 7 |
| `weight_decay` | 0.0001 |
| `optimizer` | AdamW |
| `scheduler` | Warmup + Cosine Annealing |
| `warmup_epochs` | 2 |
| `use_automixed_precision` | false |
| `loss_func` | *(to be specified)* |

<br>
<p align="center">
  <a href="https://huggingface.co/Gazeux33/VeridisQuo/tree/main" target="_blank">
    <img src="https://img.shields.io/badge/🤗%20Model-VeridisQuo-yellow?style=for-the-badge" alt="HuggingFace Model"/>
  </a>
</p>

</td>
<td width="70%" valign="top">

### Results

**Training Accuracy**
<br>
<img src="./assets/training/train_acc.png" alt="Training Accuracy" width="400">

**Training Loss**
<br>
<img src="./assets/training/train_loss.png" alt="Training Loss" width="400">

<!--**Validation Accuracy**
<br>
<img src="./assets/training/val_acc.png" alt="Validation Accuracy" width="400">

**Validation Loss**
<br>
<img src="./assets/training/val_loss.png" alt="Validation Loss" width="400">-->

</td>
</tr>
</table>


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



