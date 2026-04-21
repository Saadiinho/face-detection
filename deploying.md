# Mettre à jour et publier une nouvelle version sur PyPI

Ce guide explique comment mettre à jour correctement ton package Python et le publier sur PyPI via GitHub Actions.

---

##  1. Modifier la version du package

Dans ton fichier `pyproject.toml`, mets à jour la version :

```toml
[project]
version = "1.1.1"
````

**Important :**

* Tu dois **toujours changer la version** (ex: `1.1.0` → `1.1.1`)
* PyPI **refuse d’écraser une version existante**

---

## 2. Commit et push les modifications

```bash
git add .
git commit -m "Bump version to 1.1.1"
git push origin main
```

---

## 3. Créer un tag Git

Le workflow GitHub se déclenche via une release, donc tu dois créer un tag :

```bash
git tag v1.1.1
git push origin v1.1.1
```

---

## 4. Créer une Release GitHub

1. Va dans ton repo GitHub
2. Onglet **Releases**
3. Clique sur **"Draft a new release"**
4. Choisis le tag `v1.1.1`
5. Publie la release

Cela déclenche automatiquement le workflow GitHub Actions

---

## 5. Vérifier le déploiement

Dans GitHub :

* Onglet **Actions**
* Vérifie que le job **"Publish to PyPI"** passe au vert ✅

Sur PyPI :

* Vérifie que la nouvelle version est disponible

---

## Erreur fréquente

### `400 File already exists`

**Cause :**
Tu essaies de publier une version déjà existante.

**Solution :**

* Incrémente la version dans `pyproject.toml`
* Recommence le processus

---

## 💡 Bonnes pratiques

* Utiliser le versioning sémantique :

  * `1.1.1` → bug fix
  * `1.2.0` → nouvelle fonctionnalité
  * `2.0.0` → breaking change

* Toujours :

  * commit AVANT de créer la release
  * vérifier la version dans `pyproject.toml`

---

## Workflow résumé

```bash
# 1. Modifier version
# pyproject.toml → 1.1.1

# 2. Commit & push
git add .
git commit -m "Bump version"
git push

# 3. Tag
git tag v1.1.1
git push origin v1.1.1

# 4. Release GitHub → déclenche le publish 🚀
```

---

## Résultat

Ta nouvelle version est automatiquement :

* buildée
* uploadée
* disponible sur PyPI

---
##  Contact

Saad RAFIQUL - [@Saadiinho](https://github.com/Saadiinho) - saad.rafiqul1@gmail.com

