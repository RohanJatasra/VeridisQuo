# Harmonisation des noms de paramètres YAML

## Problèmes identifiés et recommandations

### 1. Chemin du modèle de détection de visage
**Problème** : `path` vs `custom_weights_path`

**État actuel** :
- inference_config.yml: `paths.models.face_detection.path`
- training_config.yml: `paths.models.face_detection.custom_weights_path`

**Recommandation** : Utiliser `path` partout
```yaml
paths:
  models:
    face_detection:
      path: models/face_detection/face_detection.pt
```

---

### 2. Répertoire de sortie
**Problème** : Structure incohérente

**État actuel** :
- inference_config.yml: `paths.output.results_dir`
- training_config.yml: `paths.output_dir`

**Recommandation** : Uniformiser avec structure imbriquée
```yaml
paths:
  output:
    dir: output  # ou output/training pour training
    models_dir: models
    logs_dir: logs  # ou logs/training pour training
```

---

### 3. Répertoire de logs
**Problème** : Niveau hiérarchique différent

**État actuel** :
- inference_config.yml: `paths.output.logs_dir`
- training_config.yml: `paths.logs_dir`

**Recommandation** : Déplacer au niveau racine de `paths`
```yaml
paths:
  logs_dir: logs  # ou logs/training, logs/inference
```

---

### 4. EfficientNet pretrained
**Problème** : `use_pretrained` vs `pretrained`

**État actuel** :
- inference_config.yml: `model_architecture.spatial_features.use_pretrained`
- training_config.yml: `model_architecture.spatial_features.pretrained`

**Recommandation** : Utiliser `pretrained` (convention PyTorch)
```yaml
model_architecture:
  spatial_features:
    pretrained: true
```

---

### 5. Structure logging
**Problème** : Configuration au mauvais niveau

**État actuel** :
- inference_config.yml: `logging.level`, `logging.format`, `logging.datefmt` (racine)
- training_config.yml: `logging.file.level`, `logging.file.format`, `logging.file.datefmt` (sous file)

**Recommandation** : Adopter structure inference (plus logique)
```yaml
logging:
  enabled: true
  level: DEBUG  # Global
  format: "..."  # Global
  datefmt: "..."  # Global
  file:
    enabled: true
    path: logs/app.log
    backup_count: 5
  console:
    enabled: true
    level: INFO  # Peut surcharger le niveau global
```

---

## Plan d'action

### Priorité 1 - Critiques (affectent la compatibilité du code)
1. ✅ Uniformiser `pretrained` (au lieu de `use_pretrained`)
2. ✅ Uniformiser `paths.models.face_detection.path`

### Priorité 2 - Importantes (améliorent la cohérence)
3. ✅ Restructurer `paths.output` et `paths.logs_dir`
4. ✅ Adopter structure logging d'inference dans training

### Priorité 3 - Nice to have
5. ⚪ Documenter les conventions de nommage
6. ⚪ Valider avec un schéma JSON/YAML

---

## Conventions générales à adopter

### Nommage des paramètres
- **Chemins de fichiers** : `*_path` (singulier)
- **Chemins de dossiers** : `*_dir` (singulier)
- **Booléens** : Pas de préfixe `use_*` sauf si ambiguïté (préférer forme directe)
- **Listes/Arrays** : Forme plurielle si applicable
- **Dimensions** : `*_size` pour tuples (width, height), `*_dim` pour scalaires

### Structure hiérarchique
```yaml
paths:
  project_root: ./
  data_dir: data
  output_dir: output
  logs_dir: logs
  models:
    model_name:
      path: ...

model_architecture:
  component_name:
    enabled: true
    param1: value1

preprocessing:
  stage_name:
    enabled: true
    param1: value1

logging:
  # Niveau global
  level: DEBUG
  format: "..."
  # Handlers spécifiques
  file:
    enabled: true
  console:
    enabled: true
```
