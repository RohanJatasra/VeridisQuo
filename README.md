# VeridisQuo



<table>
<tr>
<td width="60%" >
    
### *Where is the truth?*
    
#### **What is Veridis Quo?**

A state-of-the-art neural network designed to **detect deepfake videos** and **highlight the altered areas** in each frame using explainable AI.

---
#### **How does it work?**
VeridisQuo uses a **hybrid architecture** combining spatial and frequency-domain analysis. **EfficientNet-B4** extracts spatial features from faces, while **frequency analysis** captures artifacts invisible to the human eye. These streams merge into a unified classifier, achieving robust detection across multiple deepfake techniques.
</td>
<td width="40%" valign="middle" align="center">
<img src="./assets/logo/VeridisQuo.png" alt="VeridisQuo Logo" width="100%">
</td>
</tr>
<tr>
<td colspan="2" align="center">

### **Result**
<img src="https://github.com/user-attachments/assets/251d23d4-a466-4207-8c00-18af5e2769fc" alt="Démonstration VeridisQuo" width="720">
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

## Model Architecture

### Hybrid Detection System
```mermaid
graph LR
    Input["Input Image<br/>224×224×3"] --> Spatial["<b>Spatial Stream</b><br/>EfficientNet-B4<br/>(ImageNet)"]
    Input --> Frequency["<b>Frequency Stream</b>"]

    Frequency --> FFT["FFT Extractor<br/>8 radial bands<br/>Hann window"]
    Frequency --> DCT["DCT Extractor<br/>8×8 blocks<br/>frequency bands"]

    FFT --> FFT_Out["512-dim"]
    DCT --> DCT_Out["512-dim"]

    FFT_Out --> Fusion["Fusion MLP"]
    DCT_Out --> Fusion

    Spatial --> Spatial_Out["1792-dim"]
    Fusion --> Fusion_Out["1024-dim"]

    Spatial_Out --> Concat["Concatenate<br/>2816-dim"]
    Fusion_Out --> Concat

    Concat --> Classifier["<b>Classifier MLP</b><br/>1024 → 512 → 256"]
    Classifier --> Output["Output<br/>FAKE/REAL<br/>+ confidence"]

    style Input fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff
    style Spatial fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#10b981
    style Frequency fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#3b82f6
    style FFT fill:#0d1117,stroke:#06b6d4,color:#06b6d4
    style DCT fill:#0d1117,stroke:#06b6d4,color:#06b6d4
    style Fusion fill:#0d1117,stroke:#3b82f6,color:#3b82f6
    style Classifier fill:#0f172a,stroke:#ec4899,stroke-width:2px,color:#ec4899
    style Output fill:#1e293b,stroke:#a855f7,stroke-width:2px,color:#fff
    style Concat fill:#0d1117,stroke:#8b5cf6,color:#8b5cf6
    style FFT_Out fill:#0d1117,stroke:#64748b,color:#64748b
    style DCT_Out fill:#0d1117,stroke:#64748b,color:#64748b
    style Spatial_Out fill:#0d1117,stroke:#64748b,color:#64748b
    style Fusion_Out fill:#0d1117,stroke:#64748b,color:#64748b
```

### Model Specifications
| Specification | Value |
|:--------------|:------|
| **Total Parameters** | 25.05M params |
| **Input Size** | 224×224 RGB |
| **Output** | Binary (FAKE/REAL) + confidence |
| **Backbone** | EfficientNet-B4 (19.34M params) |
| **Frequency Module** | 2.16M params |
| **Classifier** | 3.54M params |



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

### Preprocessing Pipeline
The dataset was built using the following pipeline:
1. **Frame Extraction**: 1 FPS from videos (PyAV GPU-accelerated)
2. **Face Detection**: YOLO-based detector (confidence ≥ 0.7)
3. **Face Extraction**: 224×224 crops with 20px padding
4. **Dataset Split**: Stratified 70/15/15 split
5. **Class Balancing**: Oversample minority class



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
| `loss_func` | CrossEntropyLoss |

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

## API Reference

### Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/api/v1/health` | Health check and model status |
| `POST` | `/api/v1/analyze` | Analyze video for deepfakes |
| `GET` | `/api/v1/outputs/{filename}` | Download GradCAM visualization |
| `DELETE` | `/api/v1/outputs/{filename}` | Delete output file |

### Request Format (POST /api/v1/analyze)

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@video.mp4" \
  -F "fps=1" \
  -F "aggregation_method=majority" \
  -F "generate_gradcam=true"
```

**Parameters**:
- `file`: Video file (MP4, AVI, MOV, MKV, WEBM)
- `fps`: Frames per second to extract (default: 1)
- `aggregation_method`: Score aggregation (default: majority)
- `generate_gradcam`: Generate visualization video (default: false)

### Response Format

```json
{
  "prediction": "FAKE",
  "confidence": 0.8734,
  "aggregation_method": "majority",
  "total_frames": 120,
  "gradcam_video_path": "/api/v1/outputs/gradcam_video_20250102.mp4"
}
```

---



## Quick Start

### Prerequisites
- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- Node.js 18+ and npm (optional, for frontend)
- CUDA 11.8+ (optional, for GPU acceleration)

### Clone project

```bash
# Clone repository
git clone https://github.com/VeridisQuo-orga/VeridisQuo.git
cd VeridisQuo
```

### Launch backend

```bash
chmod +x ./scripts/launch_api.sh
./scripts/launch_api.sh

```

server runs on `http://localhost:8000` | Docs at `/docs`

### Launch frontend

```bash
chmod +x ./scripts/launch_frontend.sh
./scripts/launch_frontend.sh
```
Development server on `http://localhost:3000`

<p align="center">
<img src="./assets/results/frontend.png" alt="frontend" width="520">
<p/>

---



## Citation

If you use VeridisQuo in your research, please cite:

```bibtex
@software{veridisquo2025,
  title = {VeridisQuo: Hybrid Deepfake Detection with Explainable AI},
  author = {Castillo, Theo and Barriere, Clement},
  year = {2025},
  url = {https://github.com/VeridisQuo-orga/VeridisQuo},
  note = {Model: \url{https://huggingface.co/Gazeux33/VeridisQuo}}
}
```




