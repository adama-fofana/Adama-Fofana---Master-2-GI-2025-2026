markdown
# Projet d'Analyse Numérique - Master 2 Génie Informatique

## 📋 Description du Projet

Ce projet académique présente une étude comparative approfondie des méthodes numériques fondamentales en analyse numérique, réalisé dans le cadre du Master 2 Génie Informatique à l'Université Nangui Abrogoua.

Le projet se divise en deux volets principaux :
1. **Résolution numérique d'équations différentielles ordinaires (EDO)**
2. **Méthodes d'intégration numérique (quadratures)**

## 🎯 Objectifs du Projet

- Implémenter et comparer différentes méthodes numériques
- Analyser leurs performances en termes de précision et de temps d'exécution
- Fournir des recommandations pratiques pour le choix des méthodes
- Générer des rapports LaTeX professionnels pour la présentation des résultats

## 📁 Structure du Dépôt
Adama-Fofana---Master-2-GI-2025-2026/
│
├── 📄 README.md # Ce fichier
├── 📊 presentationECD-latex.txt # Présentation LaTeX EDO
├── 📊 presentationLatexIntégrationNumérique.txt # Présentation LaTeX Intégration
├── 📄 presentation_projet_analyse_numeriqueM2GI_2025_2026pdf.pdf
├── 📄 prsentation-IntégrationNumériquePDF.pdf
├── 📄 prsentation-Équations-Différentiellespdf.pdf
├── 🐍 resolution-equations-differentielles.py
└── 🐍 resolutionDintegrale.py

text

## 🔧 Implémentations Python

### 1. Résolution d'Équations Différentielles (`resolution-equations-differentielles.py`)

**Méthodes implémentées :**
- Méthode d'Euler explicite (ordre 1)
- Méthode de Heun (Euler amélioré, ordre 2)
- Méthode de Runge-Kutta d'ordre 4 (RK4)

**Équations tests :**
1. Croissance exponentielle modifiée : `z'(x) = 0.1 * x * z(x)`
2. Équation avec singularité : `z'(x) = (1 - 30x)/(2√x) + 15z(x)`
3. Coefficient périodique : `z'(x) = πcos(πx)z(x)`

### 2. Méthodes d'Intégration Numérique (`resolutionDintegrale.py`)

**Méthodes implémentées :**
- Quadrature de Gauss-Legendre
- Quadrature de Gauss-Laguerre (pour poids `e^{-x}`)
- Quadrature de Gauss-Chebyshev (pour poids `1/√(1-x²)`)
- Méthode composite de Simpson
- Intégration par spline cubique

**Fonctions tests :**
1. Fonction Chebyshev : `cos(10x)` sur [-1, 1] avec poids `1/√(1-x²)`
2. Fonction Laguerre : `1/(1 + x²)` sur [0, ∞) avec poids `e^{-x}`
3. Fonction combinée : `cos(x)` sur [0, 1] avec poids `1/√(1-x²)`
4. Fonction de Runge : `1/(1 + 25x²)` sur [-1, 1] (sans poids)

## 📊 Résultats Principaux

### Pour les Équations Différentielles :
- **RK4** : Meilleure précision (erreur ~10⁻⁶)
- **Heun** : Bon compromis précision/temps
- **Euler** : Plus rapide mais moins précise

### Pour l'Intégration Numérique :
- **Gauss-Laguerre** : Convergence exponentielle sur sa fonction cible
- **Gauss-Legendre** : Excellente pour fonctions standards
- **Simpson** : Robustesse et simplicité efficaces

## 📈 Visualisations Incluses

Le projet génère automatiquement :
- Graphiques comparatifs des solutions numériques
- Évolution des erreurs absolues
- Comparaison des temps d'exécution
- Graphes de convergence des méthodes

## 🛠️ Technologies Utilisées

- **Python 3.x** - Langage de programmation principal
- **NumPy** - Calcul scientifique et manipulation de tableaux
- **Matplotlib** - Visualisation des résultats
- **SciPy** - Fonctions scientifiques spécialisées
- **LaTeX** - Génération de rapports professionnels


# Méthodes d'intégration numérique
python resolutionDintegrale.py
📚 Documentation et Rapports
Rapports PDF Générés :
prsentation-Équations-Différentiellespdf.pdf - Analyse complète des EDO

prsentation-IntégrationNumériquePDF.pdf - Étude des méthodes d'intégration

presentation_projet_analyse_numeriqueM2GI_2025_2026pdf.pdf - Synthèse globale

Contenu des Rapports :
Présentation théorique des méthodes

Résultats expérimentaux détaillés

Analyse comparative quantitative

Recommandations pratiques

Code source commenté

Bibliographie complète

🎓 Contexte Académique
Université : Université Nangui Abrogoua
Formation : Master 2 Génie Informatique
Année académique : 2025-2026
Encadrement : Présenté devant le Docteur Sylvain ZEZE
Date de soutenance : 6 Janvier 2026

👤 Auteur
Adama Fofana

Matricule : CI0221058471

Master 2 Génie Informatique

Université Nangui Abrogoua

Email : adama5.fofana@uvci.edu.ci

