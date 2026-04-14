```markdown
# Niyya Face Detector

[![PyPI](https://img.shields.io/pypi/v/niyya-face-detector.svg)](https://pypi.org/project/niyya-face-detector/)
[![Python Versions](https://img.shields.io/pypi/pyversions/niyya-face-detector.svg)](https://pypi.org/project/niyya-face-detector/)
[![License](https://img.shields.io/pypi/l/niyya-face-detector.svg)](https://github.com/Saadiinho/niyya-face-detector/blob/main/LICENSE)
[![CI](https://github.com/Saadiinho/niyya-face-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/Saadiinho/niyya-face-detector/actions)

> Module de détection faciale pour la modération de contenu image.  
> Supporte Haar Cascades, DNN et RetinaFace via une interface unifiée.

---

## 🚀 Installation

```bash
# Installation de base
pip install niyya-face-detector

```

### Prérequis système (Linux)

```bash
# Pour OpenCV
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

---

## 📖 Usage Rapide

### Détection depuis un fichier

```python
from niyya_face_detector import FaceDetector

# Initialisation
detector = FaceDetector(model_type="retinaface")  # ou "haar", "dnn"

# Analyse d'une image
result = detector.analyze("photo.jpg")

print(f"Visage détecté: {result['has_face']}")
print(f"Nombre de visages: {result['face_count']}")
print(f"Confiance: {result['confidence']:.2%}")
```


### Analyse depuis des bytes (pour API)

```python
with open("photo.jpg", "rb") as f:
    image_bytes = f.read()

result = detector.analyze_bytes(image_bytes)
```

---

## ⚙️ Modèles Disponus

| Modèle                            | Précision | Vitesse | Usage Recommandé |
|-----------------------------------|-----------|---------|-----------------|
| `haar`                            | ⭐⭐ | ⚡⚡ | Tests rapides, prototype |
| `dnn` (en cours d'implémentation) | ⭐⭐⭐⭐ | ⚡⚡ | Production légère |
| `retinaface`                      | ⭐⭐⭐⭐⭐ | ⚡ | Hijab, occlusions, production |

> **Conseil** : Pour une détection optimale sur des visages avec occlusions partielles, utilisez le modèle `retinaface`.

---

## 📦 Options d'Installation

```bash
# Développement
pip install niyya-face-detector[dev]

# Documentation
pip install niyya-face-detector[docs]

# Tout en un
pip install niyya-face-detector[all]
```

---

## 🧪 Tests

```bash
# Installer les dépendances de test
pip install niyya-face-detector[dev]

# Lancer les tests
pytest tests/ -v
```

---

## 📚 Documentation Complète

La documentation complète est disponible sur :  
🔗 https://saadiinho.github.io/niyya-face-detector/

---

## 🤝 Contributing

Les contributions sont les bienvenues !

1. Fork le projet
2. Crée une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit tes changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvre une Pull Request

---

## 📄 Licence

Distribué sous la licence MIT. Voir [`LICENSE`](LICENSE) pour plus d'informations.

---

## 🙏 Remerciements

- [OpenCV](https://opencv.org/) pour la vision par ordinateur
- [InsightFace](https://github.com/deepinsight/insightface) pour RetinaFace
- La communauté Niyya Women pour les retours et tests

---

## 📞 Contact

Saad RAFIQUL - [@Saadiinho](https://github.com/Saadiinho) - saad.rafiqul1@gmail.com

Projet : https://github.com/Saadiinho/niyya-face-detector
