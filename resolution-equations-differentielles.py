"""
==================================================
CODE POUR LA COMPARAISON DES MÉTHODES DE RÉSOLUTION 
D'ÉQUATIONS DIFFÉRENTIELLES ORDINAIRES (EDO)
Auteur: Fofana Adama
Master 2 Génie Informatique - Université Nangui Abrogoua
Cours: Calcul Numérique / Analyse Numérique
==================================================

Ce code implémente et compare 3 méthodes de résolution d'EDO :
1. Euler explicite - Méthode la plus simple (ordre 1)
2. Heun (Euler amélioré) - Méthode de prédiction-corréction (ordre 2)
3. Runge-Kutta 4 - Méthode la plus précise (ordre 4)

Les méthodes sont testées sur 3 équations différentielles avec solutions exactes connues.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from math import sqrt, sin, cos, pi, exp

# ============================================================================
# CLASSE PRINCIPALE : SOLVEUR D'ÉQUATIONS DIFFÉRENTIELLES
# ============================================================================
class SolveurEDO:
    """
    Classe principale qui implémente les différentes méthodes de résolution d'EDO.
    Une EDO du premier ordre s'écrit : dy/dx = f(x, y), avec y(x₀) = y₀
    """
    
    def __init__(self):
        """
        Constructeur de la classe.
        Pas d'initialisation particulière nécessaire.
        """
        pass
    
    # ==========================================================================
    # 1. MÉTHODE D'EULER EXPLICITE
    # ==========================================================================
    def euler(self, f, x0, y0, h, n):
        """
        Méthode d'Euler explicite - La plus simple et intuitive
        
        PRINCIPE MATHÉMATIQUE :
        y_{n+1} = y_n + h * f(x_n, y_n)
        
        CARACTÉRISTIQUES :
        - Ordre 1 : erreur proportionnelle à h (O(h))
        - Simple et rapide à calculer
        - Peu précis pour des pas grands
        - Condition de stabilité restrictive
        
        ARGUMENTS :
        -----------
        f : fonction dérivée f(x, y) = dy/dx
        x0 : valeur initiale de x
        y0 : valeur initiale y(x0)
        h : pas d'intégration (taille du pas)
        n : nombre d'itérations
        
        RETOUR :
        --------
        x, y : tableaux numpy contenant les points calculés
        """
        # Initialisation des tableaux pour stocker les résultats
        x = np.zeros(n+1)  # n+1 points (du point 0 au point n)
        y = np.zeros(n+1)
        
        # Conditions initiales
        x[0] = x0
        y[0] = y0
        
        # BOUCLE PRINCIPALE D'INTÉGRATION
        for i in range(n):
            # Formule d'Euler : y_{i+1} = y_i + h * f(x_i, y_i)
            y[i+1] = y[i] + h * f(x[i], y[i])
            
            # Avancement de x
            x[i+1] = x[i] + h
        
        return x, y
    
    # ==========================================================================
    # 2. MÉTHODE DE HEUN (EULER AMÉLIORÉ)
    # ==========================================================================
    def heun(self, f, x0, y0, h, n):
        """
        Méthode de Heun - Méthode de prédiction-corréction (Runge-Kutta d'ordre 2)
        
        PRINCIPE MATHÉMATIQUE :
        k1 = f(x_n, y_n)
        k2 = f(x_n + h, y_n + h*k1)
        y_{n+1} = y_n + h/2 * (k1 + k2)
        
        CARACTÉRISTIQUES :
        - Ordre 2 : erreur proportionnelle à h² (O(h²))
        - Meilleure précision qu'Euler pour même pas
        - Un peu plus coûteux en calculs
        - Méthode à un pas (nécessite seulement y_n)
        """
        # Initialisation
        x = np.zeros(n+1)
        y = np.zeros(n+1)
        x[0] = x0
        y[0] = y0
        
        # BOUCLE PRINCIPALE
        for i in range(n):
            # ÉTAPE 1 : Prédiction (méthode d'Euler)
            k1 = f(x[i], y[i])
            
            # ÉTAPE 2 : Évaluation au point prédit
            k2 = f(x[i] + h, y[i] + h * k1)
            
            # ÉTAPE 3 : Correction (moyenne des pentes)
            y[i+1] = y[i] + h * (k1 + k2) / 2
            
            # Avancement de x
            x[i+1] = x[i] + h
        
        return x, y
    
    # ==========================================================================
    # 3. MÉTHODE DE RUNGE-KUTTA D'ORDRE 4 (RK4)
    # ==========================================================================
    def runge_kutta_4(self, f, x0, y0, h, n):
        """
        Méthode de Runge-Kutta d'ordre 4 - La plus utilisée en pratique
        
        PRINCIPE MATHÉMATIQUE :
        k1 = f(x_n, y_n)
        k2 = f(x_n + h/2, y_n + h/2*k1)
        k3 = f(x_n + h/2, y_n + h/2*k2)
        k4 = f(x_n + h, y_n + h*k3)
        y_{n+1} = y_n + h/6*(k1 + 2k2 + 2k3 + k4)
        
        CARACTÉRISTIQUES :
        - Ordre 4 : erreur proportionnelle à h⁴ (O(h⁴))
        - Très précis même avec des pas relativement grands
        - Quatre évaluations de fonction par pas
        - Standard industriel pour la plupart des problèmes
        """
        # Initialisation
        x = np.zeros(n+1)
        y = np.zeros(n+1)
        x[0] = x0
        y[0] = y0
        
        # BOUCLE PRINCIPALE
        for i in range(n):
            # Évaluation des 4 pentes (méthode à 4 étages)
            k1 = f(x[i], y[i])                          # Pente au début
            k2 = f(x[i] + h/2, y[i] + h*k1/2)           # Pente au milieu (avec k1)
            k3 = f(x[i] + h/2, y[i] + h*k2/2)           # Pente au milieu (avec k2)
            k4 = f(x[i] + h, y[i] + h*k3)               # Pente à la fin
            
            # Combinaison pondérée des pentes
            y[i+1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
            
            # Avancement de x
            x[i+1] = x[i] + h
        
        return x, y
    
    # ==========================================================================
    # 4. CALCUL DES ERREURS
    # ==========================================================================
    def calculer_erreur(self, y_numerique, y_exacte):
        """
        Calcule les erreurs entre solution numérique et solution exacte
        
        TYPES D'ERREUR :
        - Erreur absolue : |y_num - y_exact|
        - Erreur relative : |y_num - y_exact| / |y_exact| (évite division par 0)
        
        UTILISATION :
        Permet de quantifier la précision de chaque méthode.
        """
        # Erreur absolue (en valeur absolue)
        erreur_absolue = np.abs(y_numerique - y_exacte)
        
        # Erreur relative (avec protection contre division par 0)
        # Le terme 1e-10 évite les divisions par 0 quand y_exacte ≈ 0
        erreur_relative = np.abs(y_numerique - y_exacte) / (np.abs(y_exacte) + 1e-10)
        
        return erreur_absolue, erreur_relative


# ============================================================================
# DÉFINITION DES ÉQUATIONS DIFFÉRENTIELLES TEST
# ============================================================================
# Chaque équation est choisie pour tester différents aspects des méthodes

# ----------------------------------------------------------------------------
# ÉQUATION 1 : Croissance exponentielle modifiée
# ----------------------------------------------------------------------------
def f1(x, z):
    """
    Équation 1 : z′(x) = 0.1 * x * z(x)
    
    CARACTÉRISTIQUES :
    - Croissance plus que exponentielle (terme x*z)
    - Solution analytique connue
    - Teste la précision sur une croissance rapide
    - Pas de singularité
    """
    return 0.1 * x * z

def solution_exacte1(x):
    """
    Solution exacte de l'équation 1 : z′(x) = 0.1 * x * z(x), z(0) = 1
    
    SOLUTION :
    z(x) = exp(0.05 * x²)
    
    DÉMONSTRATION :
    dz/dx = 0.1 x z  →  dz/z = 0.1 x dx
    Intégration : ln|z| = 0.05 x² + C
    Avec z(0)=1 : C = 0, donc z = exp(0.05 x²)
    """
    return np.exp(0.05 * x**2)

# ----------------------------------------------------------------------------
# ÉQUATION 2 : Équation avec singularité et terme source
# ----------------------------------------------------------------------------
def f2(x, z):
    """
    Équation 2 : z′(x) = (1 - 30x)/(2√x) + 15z(x)
    
    CARACTÉRISTIQUES :
    - Singularité en x=0 (division par √x)
    - Terme source dépendant de x
    - Terme linéaire en z
    - Solution connue : √x
    - Teste la gestion des singularités
    """
    if x == 0:
        return 0  # Gestion de la singularité
    return (1 - 30*x) / (2*sqrt(x)) + 15*z

def solution_exacte2(x):
    """
    Solution exacte de l'équation 2 : z(x) = √x
    
    VÉRIFICATION :
    z = √x → z′ = 1/(2√x)
    RHS = (1 - 30x)/(2√x) + 15√x = 1/(2√x) - 30x/(2√x) + 15√x
         = 1/(2√x) - 15√x + 15√x = 1/(2√x) = z′ ✓
    """
    return np.sqrt(x)

# ----------------------------------------------------------------------------
# ÉQUATION 3 : Équation à coefficient périodique
# ----------------------------------------------------------------------------
def f3(x, z):
    """
    Équation 3 : z′(x) = π cos(πx) z(x)
    
    CARACTÉRISTIQUES :
    - Coefficient périodique (cos(πx))
    - Croissance/décroissance alternée
    - Solution oscillante amortie/exponentielle
    - Teste le comportement avec coefficients variables
    """
    return pi * cos(pi * x) * z

def solution_exacte3(x):
    """
    Solution exacte de l'équation 3 : z(x) = exp(sin(πx))
    
    DÉMONSTRATION :
    dz/dx = π cos(πx) z
    dz/z = π cos(πx) dx
    Intégration : ln|z| = sin(πx) + C
    Avec z(0)=1 : C = 0, donc z = exp(sin(πx))
    """
    return np.exp(np.sin(pi * x))


# ============================================================================
# FONCTION PRINCIPALE DE TEST
# ============================================================================
def tester_methodes():
    """
    Fonction principale qui compare les 3 méthodes sur les 3 équations
    
    ORGANISATION :
    1. Création du solveur
    2. Tests successifs sur chaque équation
    3. Calcul des temps d'exécution
    4. Calcul des erreurs
    5. Visualisation des résultats
    """
    
    # Initialisation du solveur
    solveur = SolveurEDO()
    
    # Paramètre commun : pas d'intégration
    # Un pas plus petit donnerait de meilleures précisions mais plus de calculs
    pas = 0.3
    
    # ======================================================================
    # TEST 1 : ÉQUATION 1 (croissance exponentielle modifiée)
    # ======================================================================
    print("=" * 70)
    print("ÉQUATION 1: z′(x) = 0.1 * x * z(x) avec z(0) = 1")
    print("=" * 70)
    
    # Paramètres pour l'équation 1
    x0, y0 = 0, 1      # Condition initiale
    b = 5              # Borne supérieure d'intégration
    n = int((b - x0) / pas)  # Nombre d'itérations
    
    # ------------------------------------------------------------------
    # RÉSOLUTION AVEC LES 3 MÉTHODES (avec chronométrage)
    # ------------------------------------------------------------------
    
    # Méthode d'Euler
    temps_debut = time.time()
    x_euler1, y_euler1 = solveur.euler(f1, x0, y0, pas, n)
    temps_euler1 = time.time() - temps_debut
    
    # Méthode de Heun
    temps_debut = time.time()
    x_heun1, y_heun1 = solveur.heun(f1, x0, y0, pas, n)
    temps_heun1 = time.time() - temps_debut
    
    # Méthode Runge-Kutta 4
    temps_debut = time.time()
    x_rk41, y_rk41 = solveur.runge_kutta_4(f1, x0, y0, pas, n)
    temps_rk41 = time.time() - temps_debut
    
    # Solution exacte (évaluée aux mêmes points qu'Euler pour comparaison)
    y_exacte1 = solution_exacte1(x_euler1)
    
    # ------------------------------------------------------------------
    # CALCUL DES ERREURS POUR L'ÉQUATION 1
    # ------------------------------------------------------------------
    erreur_abs_euler1, erreur_rel_euler1 = solveur.calculer_erreur(y_euler1, y_exacte1)
    erreur_abs_heun1, erreur_rel_heun1 = solveur.calculer_erreur(y_heun1, y_exacte1)
    erreur_abs_rk41, erreur_rel_rk41 = solveur.calculer_erreur(y_rk41, y_exacte1)
    
    # ------------------------------------------------------------------
    # AFFICHAGE DES RÉSULTATS POUR L'ÉQUATION 1
    # ------------------------------------------------------------------
    print(f"\nTemps d'exécution - Équation 1:")
    print(f"Euler: {temps_euler1:.6f} secondes")
    print(f"Heun: {temps_heun1:.6f} secondes")
    print(f"Runge-Kutta 4: {temps_rk41:.6f} secondes")
    
    print(f"\nErreur absolue maximale - Équation 1:")
    print(f"Euler: {np.max(erreur_abs_euler1):.6e}")
    print(f"Heun: {np.max(erreur_abs_heun1):.6e}")
    print(f"Runge-Kutta 4: {np.max(erreur_abs_rk41):.6e}")
    
    # ======================================================================
    # TEST 2 : ÉQUATION 2 (avec singularité)
    # ======================================================================
    print("\n" + "=" * 70)
    print("ÉQUATION 2: z′(x) = (1 - 30x)/(2√x) + 15z(x) avec z(0)=0")
    print("Solution exacte: z(x) = √x")
    print("=" * 70)
    
    # Paramètres pour l'équation 2 (éviter la singularité en x=0)
    x0, y0 = 0.01, sqrt(0.01)  # On démarre légèrement après 0
    b = 2
    n = int((b - x0) / pas)
    
    # Résolution avec les 3 méthodes
    temps_debut = time.time()
    x_euler2, y_euler2 = solveur.euler(f2, x0, y0, pas, n)
    temps_euler2 = time.time() - temps_debut
    
    temps_debut = time.time()
    x_heun2, y_heun2 = solveur.heun(f2, x0, y0, pas, n)
    temps_heun2 = time.time() - temps_debut
    
    temps_debut = time.time()
    x_rk42, y_rk42 = solveur.runge_kutta_4(f2, x0, y0, pas, n)
    temps_rk42 = time.time() - temps_debut
    
    # Solution exacte
    y_exacte2 = solution_exacte2(x_euler2)
    
    # Calcul des erreurs
    erreur_abs_euler2, erreur_rel_euler2 = solveur.calculer_erreur(y_euler2, y_exacte2)
    erreur_abs_heun2, erreur_rel_heun2 = solveur.calculer_erreur(y_heun2, y_exacte2)
    erreur_abs_rk42, erreur_rel_rk42 = solveur.calculer_erreur(y_rk42, y_exacte2)
    
    # Affichage
    print(f"\nTemps d'exécution - Équation 2:")
    print(f"Euler: {temps_euler2:.6f} secondes")
    print(f"Heun: {temps_heun2:.6f} secondes")
    print(f"Runge-Kutta 4: {temps_rk42:.6f} secondes")
    
    print(f"\nErreur absolue maximale - Équation 2:")
    print(f"Euler: {np.max(erreur_abs_euler2):.6e}")
    print(f"Heun: {np.max(erreur_abs_heun2):.6e}")
    print(f"Runge-Kutta 4: {np.max(erreur_abs_rk42):.6e}")
    
    # ======================================================================
    # TEST 3 : ÉQUATION 3 (coefficient périodique)
    # ======================================================================
    print("\n" + "=" * 70)
    print("ÉQUATION 3: z′(x) = πcos(πx)z(x) avec z(0)=0")
    print("Solution exacte: z(x) = exp(sin(πx))")
    print("Note: z(0)=exp(sin(0))=1 (correction de la condition initiale)")
    print("=" * 70)
    
    # Paramètres pour l'équation 3 (correction: z(0)=1, pas 0)
    x0, y0 = 0, 1
    b = 2
    n = int((b - x0) / pas)
    
    # Résolution avec les 3 méthodes
    temps_debut = time.time()
    x_euler3, y_euler3 = solveur.euler(f3, x0, y0, pas, n)
    temps_euler3 = time.time() - temps_debut
    
    temps_debut = time.time()
    x_heun3, y_heun3 = solveur.heun(f3, x0, y0, pas, n)
    temps_heun3 = time.time() - temps_debut
    
    temps_debut = time.time()
    x_rk43, y_rk43 = solveur.runge_kutta_4(f3, x0, y0, pas, n)
    temps_rk43 = time.time() - temps_debut
    
    # Solution exacte
    y_exacte3 = solution_exacte3(x_euler3)
    
    # Calcul des erreurs
    erreur_abs_euler3, erreur_rel_euler3 = solveur.calculer_erreur(y_euler3, y_exacte3)
    erreur_abs_heun3, erreur_rel_heun3 = solveur.calculer_erreur(y_heun3, y_exacte3)
    erreur_abs_rk43, erreur_rel_rk43 = solveur.calculer_erreur(y_rk43, y_exacte3)
    
    # Affichage
    print(f"\nTemps d'exécution - Équation 3:")
    print(f"Euler: {temps_euler3:.6f} secondes")
    print(f"Heun: {temps_heun3:.6f} secondes")
    print(f"Runge-Kutta 4: {temps_rk43:.6f} secondes")
    
    print(f"\nErreur absolue maximale - Équation 3:")
    print(f"Euler: {np.max(erreur_abs_euler3):.6e}")
    print(f"Heun: {np.max(erreur_abs_heun3):.6e}")
    print(f"Runge-Kutta 4: {np.max(erreur_abs_rk43):.6e}")
    
    # ======================================================================
    # VISUALISATION DES RÉSULTATS
    # ======================================================================
    
    # ------------------------------------------------------------------
    # FIGURE 1 : COMPARAISON MÉTHODE PAR MÉTHODE
    # ------------------------------------------------------------------
    # 3 lignes (équations) × 3 colonnes (méthodes) = 9 sous-graphiques
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Ligne 1 : Équation 1
    # Colonne 1 : Euler vs Exact
    axes[0, 0].plot(x_euler1, y_exacte1, 'k-', label='Solution exacte', linewidth=2)
    axes[0, 0].plot(x_euler1, y_euler1, 'ro-', label='Euler', markersize=4)
    axes[0, 0].set_title('Équation 1 - Euler vs Exact')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('z(x)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Colonne 2 : Heun vs Exact
    axes[0, 1].plot(x_heun1, y_exacte1, 'k-', label='Solution exacte', linewidth=2)
    axes[0, 1].plot(x_heun1, y_heun1, 'go-', label='Heun', markersize=4)
    axes[0, 1].set_title('Équation 1 - Heun vs Exact')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('z(x)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Colonne 3 : RK4 vs Exact
    axes[0, 2].plot(x_rk41, y_exacte1, 'k-', label='Solution exacte', linewidth=2)
    axes[0, 2].plot(x_rk41, y_rk41, 'bo-', label='Runge-Kutta 4', markersize=4)
    axes[0, 2].set_title('Équation 1 - RK4 vs Exact')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('z(x)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Ligne 2 : Équation 2
    axes[1, 0].plot(x_euler2, y_exacte2, 'k-', label='Solution exacte', linewidth=2)
    axes[1, 0].plot(x_euler2, y_euler2, 'ro-', label='Euler', markersize=4)
    axes[1, 0].set_title('Équation 2 - Euler vs Exact')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('z(x)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(x_heun2, y_exacte2, 'k-', label='Solution exacte', linewidth=2)
    axes[1, 1].plot(x_heun2, y_heun2, 'go-', label='Heun', markersize=4)
    axes[1, 1].set_title('Équation 2 - Heun vs Exact')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('z(x)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(x_rk42, y_exacte2, 'k-', label='Solution exacte', linewidth=2)
    axes[1, 2].plot(x_rk42, y_rk42, 'bo-', label='Runge-Kutta 4', markersize=4)
    axes[1, 2].set_title('Équation 2 - RK4 vs Exact')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('z(x)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    # Ligne 3 : Équation 3
    axes[2, 0].plot(x_euler3, y_exacte3, 'k-', label='Solution exacte', linewidth=2)
    axes[2, 0].plot(x_euler3, y_euler3, 'ro-', label='Euler', markersize=4)
    axes[2, 0].set_title('Équation 3 - Euler vs Exact')
    axes[2, 0].set_xlabel('x')
    axes[2, 0].set_ylabel('z(x)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(x_heun3, y_exacte3, 'k-', label='Solution exacte', linewidth=2)
    axes[2, 1].plot(x_heun3, y_heun3, 'go-', label='Heun', markersize=4)
    axes[2, 1].set_title('Équation 3 - Heun vs Exact')
    axes[2, 1].set_xlabel('x')
    axes[2, 1].set_ylabel('z(x)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    axes[2, 2].plot(x_rk43, y_exacte3, 'k-', label='Solution exacte', linewidth=2)
    axes[2, 2].plot(x_rk43, y_rk43, 'bo-', label='Runge-Kutta 4', markersize=4)
    axes[2, 2].set_title('Équation 3 - RK4 vs Exact')
    axes[2, 2].set_xlabel('x')
    axes[2, 2].set_ylabel('z(x)')
    axes[2, 2].legend()
    axes[2, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # ------------------------------------------------------------------
    # FIGURE 2 : ERREURS ABSOLUES
    # ------------------------------------------------------------------
    # 1 ligne × 3 colonnes = 3 sous-graphiques (un par équation)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    # Équation 1 : Erreurs
    axes2[0].plot(x_euler1, erreur_abs_euler1, 'r-', label='Euler', linewidth=2)
    axes2[0].plot(x_heun1, erreur_abs_heun1, 'g-', label='Heun', linewidth=2)
    axes2[0].plot(x_rk41, erreur_abs_rk41, 'b-', label='RK4', linewidth=2)
    axes2[0].set_title('Erreurs absolues - Équation 1')
    axes2[0].set_xlabel('x')
    axes2[0].set_ylabel('Erreur absolue')
    axes2[0].legend()
    axes2[0].grid(True)
    
    # Équation 2 : Erreurs
    axes2[1].plot(x_euler2, erreur_abs_euler2, 'r-', label='Euler', linewidth=2)
    axes2[1].plot(x_heun2, erreur_abs_heun2, 'g-', label='Heun', linewidth=2)
    axes2[1].plot(x_rk42, erreur_abs_rk42, 'b-', label='RK4', linewidth=2)
    axes2[1].set_title('Erreurs absolues - Équation 2')
    axes2[1].set_xlabel('x')
    axes2[1].set_ylabel('Erreur absolue')
    axes2[1].legend()
    axes2[1].grid(True)
    
    # Équation 3 : Erreurs
    axes2[2].plot(x_euler3, erreur_abs_euler3, 'r-', label='Euler', linewidth=2)
    axes2[2].plot(x_heun3, erreur_abs_heun3, 'g-', label='Heun', linewidth=2)
    axes2[2].plot(x_rk43, erreur_abs_rk43, 'b-', label='RK4', linewidth=2)
    axes2[2].set_title('Erreurs absolues - Équation 3')
    axes2[2].set_xlabel('x')
    axes2[2].set_ylabel('Erreur absolue')
    axes2[2].legend()
    axes2[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # ======================================================================
    # RETOUR DES RÉSULTATS POUR ANALYSE ULTERIEURE
    # ======================================================================
    return {
        'equations': [f1, f2, f3],
        'solutions': [solution_exacte1, solution_exacte2, solution_exacte3],
        'methodes': ['Euler', 'Heun', 'Runge-Kutta 4'],
        'temps': [
            [temps_euler1, temps_heun1, temps_rk41],
            [temps_euler2, temps_heun2, temps_rk42],
            [temps_euler3, temps_heun3, temps_rk43]
        ],
        'erreurs': [
            [erreur_abs_euler1, erreur_abs_heun1, erreur_abs_rk41],
            [erreur_abs_euler2, erreur_abs_heun2, erreur_abs_rk42],
            [erreur_abs_euler3, erreur_abs_heun3, erreur_abs_rk43]
        ]
    }


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    """
    POINT D'ENTRÉE DU PROGRAMME
    
    DÉROULEMENT :
    1. Affichage du titre
    2. Exécution des tests
    3. Synthèse des résultats
    4. Conclusions
    """
    
    print("COMPARAISON DES MÉTHODES DE RÉSOLUTION D'ÉQUATIONS DIFFÉRENTIELLES")
    print("=" * 70)
    
    # Étape 1 : Exécution des tests
    resultats = tester_methodes()
    
    # Étape 2 : Synthèse des résultats
    print("\n" + "=" * 70)
    print("SYNTHÈSE DES RÉSULTATS")
    print("=" * 70)
    
    for i in range(3):
        print(f"\nÉquation {i+1}:")
        print(f"  Euler - Temps: {resultats['temps'][i][0]:.6f}s, Erreur max: {np.max(resultats['erreurs'][i][0]):.6e}")
        print(f"  Heun - Temps: {resultats['temps'][i][1]:.6f}s, Erreur max: {np.max(resultats['erreurs'][i][1]):.6e}")
        print(f"  RK4 - Temps: {resultats['temps'][i][2]:.6f}s, Erreur max: {np.max(resultats['erreurs'][i][2]):.6e}")
    
    # Étape 3 : Conclusions
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("1. Runge-Kutta 4 est la méthode la plus précise (erreur minimale)")
    print("2. Euler est la méthode la plus rapide mais la moins précise")
    print("3. Heun est un bon compromis entre précision et vitesse")
    print("4. Pour des pas plus petits, toutes les méthodes seraient plus précises")
