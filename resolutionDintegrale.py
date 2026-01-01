"""
==================================================
CODE POUR LA COMPARAISON DES MÉTHODES D'INTÉGRATION NUMÉRIQUE
Auteur: Fofana Adama
Master 2 Génie Informatique - Université Nangui Abrogoua
Cours: Calcul Numérique / Analyse Numérique
==================================================

Ce code implémente et compare 5 méthodes d'intégration numérique :
1. Gauss-Legendre - Quadrature classique pour intervalles finis
2. Gauss-Laguerre - Pour intégrales sur [0,∞) avec poids e^(-x)
3. Gauss-Chebyshev - Pour intégrales avec poids 1/√(1-x²)
4. Simpson composite - Méthode classique de Simpson
5. Spline cubique - Intégration par interpolation spline

L'objectif est d'évaluer la précision, le temps d'exécution et la convergence
de ces méthodes sur différentes fonctions tests.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import integrate
from scipy.special import roots_legendre, roots_laguerre, roots_chebyt
import math
import warnings
warnings.filterwarnings('ignore')  # Ignorer certains warnings pour la lisibilité

# ============================================================================
# CLASSE PRINCIPALE : INTÉGRATION NUMÉRIQUE
# ============================================================================
class IntegrationNumerique:
    """
    Classe principale qui implémente les différentes méthodes d'intégration numérique.
    Chaque méthode est implémentée comme une fonction distincte.
    """
    
    def __init__(self):
        """
        Constructeur de la classe. 
        Pas d'initialisation particulière nécessaire pour nos méthodes.
        """
        pass
    
    # ==========================================================================
    # 1. QUADRATURE DE GAUSS-LEGENDRE
    # ==========================================================================
    def gauss_legendre(self, f, a, b, n):
        """
        Intégration par quadrature de Gauss-Legendre - Méthode la plus classique
        
        PRINCIPE MATHÉMATIQUE :
        ∫_a^b f(x)dx ≈ Σ w_i * f(x_i) après transformation linéaire
        
        ARGUMENTS :
        -----------
        f : fonction à intégrer (doit être vectorisée)
        a, b : bornes d'intégration (b peut être infini)
        n : nombre de points de Gauss (détermine la précision)
        
        RETOUR :
        --------
        float : valeur approximative de l'intégrale
        """
        # CAS PARTICULIER : Intervalle infini [a, ∞)
        if b == np.inf:
            # Transformation de variable pour ramener à un intervalle fini
            # x = t/(1-t²) avec t ∈ [-0.999, 0.999]
            def g(t):
                x = t / (1 - t**2) if t != 1 else np.inf
                return f(x) * (1 + t**2) / (1 - t**2)**2
            
            a_tr = -0.999  # Borne transformée inférieure
            b_tr = 0.999   # Borne transformée supérieure
        else:
            # CAS STANDARD : Intervalle fini
            g = f           # Pas de transformation nécessaire
            a_tr = a
            b_tr = b
        
        # ÉTAPE 1 : Obtenir les points et poids de Gauss-Legendre sur [-1, 1]
        # Ces points sont optimaux pour minimiser l'erreur d'intégration
        x, w = roots_legendre(n)
        
        # ÉTAPE 2 : Transformation linéaire de [-1, 1] vers [a_tr, b_tr]
        # Formule : t = 0.5*(b-a)*x + 0.5*(a+b)
        t = 0.5 * (b_tr - a_tr) * x + 0.5 * (a_tr + b_tr)
        
        # ÉTAPE 3 : Calcul de l'intégrale
        # Formule : ∫ ≈ 0.5*(b-a) * Σ w_i * f(t_i)
        integral = 0.5 * (b_tr - a_tr) * np.sum(w * g(t))
        
        return integral
    
    # ==========================================================================
    # 2. QUADRATURE DE GAUSS-LAGUERRE
    # ==========================================================================
    def gauss_laguerre(self, f, n):
        """
        Quadrature spécialisée pour les intégrales sur [0, ∞) avec poids e^(-x)
        
        PRINCIPE MATHÉMATIQUE :
        ∫_0^∞ e^(-x) f(x)dx ≈ Σ w_i * f(x_i)
        
        IMPORTANT : La fonction f ne doit PAS inclure le poids e^(-x)
        Les poids w_i incluent déjà ce terme exponentiel.
        
        APPLICATIONS TYPIQUES :
        - Intégrales en physique statistique
        - Transformées de Laplace
        - Problèmes de décroissance exponentielle
        """
        # ÉTAPE 1 : Points et poids optimaux pour le poids e^(-x)
        # Ces points sont les racines des polynômes de Laguerre
        x, w = roots_laguerre(n)
        
        # ÉTAPE 2 : Calcul direct de l'intégrale
        # Les poids w_i sont déjà optimisés pour le poids e^(-x)
        integral = np.sum(w * f(x))
        
        return integral
    
    # ==========================================================================
    # 3. QUADRATURE DE GAUSS-CHEBYSHEV
    # ==========================================================================
    def gauss_chebyshev(self, f, a=-1, b=1, n=10):
        """
        Quadrature spécialisée pour les intégrales avec poids 1/√(1-x²)
        
        PRINCIPE MATHÉMATIQUE :
        ∫_{-1}^1 f(x)/√(1-x²) dx ≈ Σ w_i * f(x_i)
        
        PARTICULARITÉS :
        - Très efficace pour les fonctions oscillantes
        - Les points de Chebyshev sont équidistants sur le cercle unité
        - Convergence rapide pour les fonctions analytiques
        
        UTILISATION :
        - Série de Fourier
        - Problèmes de potentiel
        - Filtrage numérique
        """
        # ÉTAPE 1 : Points et poids de Chebyshev sur [-1, 1]
        x, w = roots_chebyt(n)
        
        # ÉTAPE 2 : Adaptation à un intervalle [a, b] différent
        if a != -1 or b != 1:
            # Transformation linéaire
            t = 0.5 * (b - a) * x + 0.5 * (a + b)
            # Ajustement des poids pour la transformation
            w = w * 0.5 * (b - a)
            integral = np.sum(w * f(t))
        else:
            # Cas standard sur [-1, 1]
            integral = np.sum(w * f(x))
        
        return integral
    
    # ==========================================================================
    # 4. MÉTHODE COMPOSITE DE SIMPSON
    # ==========================================================================
    def simpson_composite(self, f, a, b, n):
        """
        Méthode composite de Simpson - Classique et facile à implémenter
        
        PRINCIPE MATHÉMATIQUE :
        Approximation parabolique sur chaque sous-intervalle
        
        FORMULE COMPOSITE :
        ∫ ≈ (h/3)[f(x₀)+f(x_n) + 4Σf(x_impairs) + 2Σf(x_pairs)]
        
        AVANTAGES :
        - Simplicité d'implémentation
        - Bon compromis précision/temps
        - Convergence de l'ordre de h⁴
        
        INCONVÉNIENTS :
        - Nécessite un nombre pair d'intervalles
        - Moins efficace que Gauss pour même nombre de points
        """
        # CAS PARTICULIER : Intervalle infini
        if b == np.inf:
            # Transformation x = t/(1-t) pour [0, ∞) → [0, 1)
            def g(t):
                x = t / (1 - t) if t != 1 else np.inf
                return f(x) / (1 - t)**2
            
            a_tr = 0
            b_tr = 0.999  # Éviter la singularité en t=1
            f_tr = g
        else:
            # CAS STANDARD
            a_tr = a
            b_tr = b
            f_tr = f
        
        # ÉTAPE 1 : Vérifier que n est pair (condition de Simpson)
        if n % 2 == 1:
            n += 1  # Correction si n est impair
        
        # ÉTAPE 2 : Calcul du pas et création des points
        h = (b_tr - a_tr) / n
        x = np.linspace(a_tr, b_tr, n+1)
        
        # ÉTAPE 3 : Évaluation de la fonction aux points
        y = f_tr(x)
        
        # ÉTAPE 4 : Application de la formule composite de Simpson
        integral = h/3 * (y[0] + y[-1] + 
                         4*np.sum(y[1:-1:2]) +  # Termes impairs
                         2*np.sum(y[2:-2:2]))   # Termes pairs
        
        return integral
    
    # ==========================================================================
    # 5. INTÉGRATION PAR SPLINE CUBIQUE
    # ==========================================================================
    def integration_spline(self, f, a, b, n):
        """
        Intégration par interpolation spline cubique
        
        PRINCIPE MATHÉMATIQUE :
        1. Échantillonner f aux points x_i
        2. Construire une spline cubique qui interpole ces points
        3. Intégrer analytiquement la spline
        
        AVANTAGES :
        - Fournit aussi une interpolation continue
        - Bonne précision pour fonctions lisses
        - Dérivable partout
        
        INCONVÉNIENTS :
        - Coûteux en temps pour grand n
        - Phénomène de Runge possible aux bords
        """
        from scipy.interpolate import CubicSpline
        
        # CAS PARTICULIER : Intervalle infini (même transformation que Simpson)
        if b == np.inf:
            def g(t):
                x = t / (1 - t) if t != 1 else np.inf
                return f(x) / (1 - t)**2
            
            a_tr = 0
            b_tr = 0.999
            f_tr = g
        else:
            a_tr = a
            b_tr = b
            f_tr = f
        
        # ÉTAPE 1 : Création des points d'interpolation équidistants
        x = np.linspace(a_tr, b_tr, n)
        
        # ÉTAPE 2 : Évaluation de la fonction
        y = f_tr(x)
        
        # ÉTAPE 3 : Construction de la spline cubique
        # CubicSpline utilise des conditions aux limites naturelles
        cs = CubicSpline(x, y)
        
        # ÉTAPE 4 : Intégration analytique de la spline
        # La méthode integrate() calcule l'intégrale exacte des polynômes cubiques
        integral = cs.integrate(a_tr, b_tr)
        
        return integral


# ============================================================================
# CLASSE DES FONCTIONS TESTS
# ============================================================================
class FonctionsTests:
    """
    Classe contenant les fonctions tests et leurs valeurs exactes.
    Chaque fonction est choisie pour mettre en évidence les forces/faiblesses
    des différentes méthodes.
    """
    
    @staticmethod
    def f1_chebyshev(x):
        """
        FONCTION CHEBYSHEV - Test pour Gauss-Chebyshev
        
        f(x) = cos(10x) sur [-1, 1] avec poids 1/√(1-x²)
        
        CARACTÉRISTIQUES :
        - Fortement oscillante (10 périodes sur [-1,1])
        - Poids singulier aux bornes
        - Représente bien les problèmes de série de Fourier
        """
        return np.cos(10 * x)
    
    @staticmethod
    def f1_exacte():
        """
        Valeur exacte de l'intégrale Chebyshev
        
        FORMULE CONNUE :
        ∫_{-1}^{1} cos(10x)/√(1-x²) dx = π * J₀(10)
        
        où J₀ est la fonction de Bessel de première espèce d'ordre 0.
        Cette formule montre le lien entre intégrales de Chebyshev et fonctions de Bessel.
        """
        from scipy.special import j0
        return math.pi * j0(10)
    
    @staticmethod
    def f2_laguerre(x):
        """
        FONCTION LAGUERRE - Test pour Gauss-Laguerre
        
        f(x) = 1/(1 + x²) sur [0, ∞) avec poids e^(-x)
        
        CARACTÉRISTIQUES :
        - Décroissance lente (comme 1/x²)
        - Poids exponentiel e^(-x)
        - Représente des problèmes de relaxation ou de décroissance
        """
        return 1 / (1 + x**2)
    
    @staticmethod
    def f2_exacte():
        """
        Valeur exacte de l'intégrale Laguerre
        
        FORMULE CONNUE :
        ∫_{0}^{∞} e^(-x)/(1 + x²) dx = Ci(1)sin(1) + (π/2 - Si(1))cos(1)
        
        où Si et Ci sont les intégrales sinus et cosinus.
        Cette valeur peut être calculée analytiquement via les fonctions spéciales.
        """
        from scipy.special import sici
        Si, Ci = sici(1)  # Calcul des intégrales sinus et cosinus en 1
        return Ci * np.sin(1) + (np.pi/2 - Si) * np.cos(1)
    
    @staticmethod
    def f3_combinee(x):
        """
        FONCTION COMBINÉE - Test mixte
        
        f(x) = cos(x) sur [0, 1) avec poids 1/√(1-x²)
        
        CARACTÉRISTIQUES :
        - Combinaison des singularités Chebyshev
        - Fonction oscillante modérée
        - Teste l'adaptabilité des méthodes
        """
        return np.cos(x)
    
    @staticmethod
    def f3_exacte():
        """
        Valeur exacte de l'intégrale combinée
        
        FORMULE :
        ∫_{0}^{1} cos(x)/√(1-x²) dx = (π/2) * J₀(1)
        
        Similaire à f1 mais avec argument différent.
        """
        from scipy.special import j0
        return np.pi/2 * j0(1)
    
    @staticmethod
    def f4_neutre(x):
        """
        FONCTION NEUTRE - Test standard (sans poids)
        
        f(x) = 1/(1 + 25x²) sur [-1, 1] (fonction de Runge)
        
        CARACTÉRISTIQUES :
        - Pas de poids particulier
        - Problème classique de Runge (instabilité d'interpolation)
        - Fonction lisse mais avec dérivées importantes aux bords
        """
        return 1 / (1 + 25 * x**2)
    
    @staticmethod
    def f4_exacte():
        """
        Valeur exacte de l'intégrale neutre
        
        CALCUL DIRECT :
        ∫_{-1}^{1} 1/(1 + 25x²) dx = (2/5) * arctan(5)
        
        Intégrale élémentaire qui peut être calculée par changement de variable.
        """
        return (2/5) * np.arctan(5)


# ============================================================================
# FONCTION PRINCIPALE DE TEST
# ============================================================================
def tester_methodes():
    """
    Fonction principale qui compare toutes les méthodes sur toutes les fonctions
    
    ORGANISATION :
    1. Création des objets d'intégration et de fonctions tests
    2. Configuration des paramètres de test
    3. Boucles sur les fonctions et les méthodes
    4. Calcul des erreurs et temps d'exécution
    5. Stockage des résultats pour analyse
    """
    # Initialisation des objets
    integ = IntegrationNumerique()  # Objet d'intégration
    funcs = FonctionsTests()        # Objet des fonctions tests
    
    # Dictionnaire des méthodes à tester
    # Format : {'Nom méthode': fonction_implémentée}
    methodes = {
        'Gauss-Legendre': integ.gauss_legendre,
        'Gauss-Laguerre': integ.gauss_laguerre,
        'Gauss-Chebyshev': integ.gauss_chebyshev,
        'Simpson': integ.simpson_composite,
        'Spline': integ.integration_spline
    }
    
    # Configuration détaillée des fonctions tests
    # Chaque entrée contient tous les paramètres nécessaires
    fonctions_config = [
        {
            'nom': 'Chebyshev',
            'f': funcs.f1_chebyshev,
            'exacte': funcs.f1_exacte(),
            'intervalle': (-1, 1),
            'methode_speciale': 'Gauss-Chebyshev',
            'poids': '1/√(1-x²)'
        },
        {
            'nom': 'Laguerre',
            'f': funcs.f2_laguerre,
            'exacte': funcs.f2_exacte(),
            'intervalle': (0, np.inf),
            'methode_speciale': 'Gauss-Laguerre',
            'poids': 'e^{-x}'
        },
        {
            'nom': 'Combinee',
            'f': funcs.f3_combinee,
            'exacte': funcs.f3_exacte(),
            'intervalle': (0, 1),
            'methode_speciale': 'Gauss-Chebyshev',
            'poids': '1/√(1-x²)'
        },
        {
            'nom': 'Neutre',
            'f': funcs.f4_neutre,
            'exacte': funcs.f4_exacte(),
            'intervalle': (-1, 1),
            'methode_speciale': None,
            'poids': 'aucun'
        }
    ]
    
    # Valeurs de n à tester : progression géométrique
    # n représente le nombre de points/maille
    n_values = [4, 8, 16, 32, 64, 128]
    
    # Structure pour stocker les résultats
    # Format : resultats[nom_fonction][nom_méthode][type_donnée]
    resultats = {func['nom']: {meth: {'erreurs': [], 'temps': [], 'n': []} 
                              for meth in methodes.keys()} 
                for func in fonctions_config}
    
    # Nombre de répétitions pour moyenner le temps d'exécution
    # Important car le temps CPU peut varier
    repetitions = 5
    
    # En-tête du programme
    print("\n" + "="*80)
    print("COMPARAISON DES MÉTHODES D'INTÉGRATION NUMÉRIQUE")
    print("="*80)
    
    # ======================================================================
    # BOUCLE PRINCIPALE SUR LES FONCTIONS
    # ======================================================================
    for func_config in fonctions_config:
        nom = func_config['nom']
        f = func_config['f']
        exacte = func_config['exacte']
        a, b = func_config['intervalle']
        methode_speciale = func_config['methode_speciale']
        
        # Affichage des informations sur la fonction testée
        print(f"\n{'='*70}")
        print(f"FONCTION: {nom}")
        print(f"Intervalle: [{a}, {b}]")
        print(f"Poids: {func_config['poids']}")
        print(f"Valeur exacte: {exacte:.10e}")
        print(f"{'='*70}")
        
        # ==================================================================
        # BOUCLE SUR LES MÉTHODES
        # ==================================================================
        for meth_name, meth_func in methodes.items():
            print(f"\n  {meth_name}:")
            
            # Initialisation des listes pour cette méthode
            erreurs = []    # Erreurs absolues
            temps_moy = []  # Temps moyens
            n_vals = []     # Valeurs de n testées
            
            # ==============================================================
            # BOUCLE SUR LES VALEURS DE n
            # ==============================================================
            for n in n_values:
                try:
                    # Initialisation
                    temps_total = 0
                    val = None
                    
                    # Moyennage sur plusieurs exécutions
                    for _ in range(repetitions):
                        start = time.perf_counter()  # Chronomètre précis
                        
                        # GESTION DES CAS PARTICULIERS
                        # ============================
                        
                        # Cas 1: Gauss-Laguerre (nécessite b=∞)
                        if meth_name == 'Gauss-Laguerre':
                            if b == np.inf:
                                val = meth_func(f, n)  # Signature spéciale
                            else:
                                val = None  # Non applicable
                                
                        # Cas 2: Gauss-Chebyshev avec poids
                        elif meth_name == 'Gauss-Chebyshev' and methode_speciale == 'Gauss-Chebyshev':
                            # Pour Chebyshev, il faut multiplier par sqrt(1-x²)
                            # car la méthode inclut déjà le poids 1/√(1-x²)
                            def f_chebyshev(x):
                                return f(x) * np.sqrt(1 - x**2)
                            val = meth_func(f_chebyshev, a, b, n)
                            
                        # Cas 3: Gauss-Chebyshev sur intervalle infini
                        elif meth_name == 'Gauss-Chebyshev' and b == np.inf:
                            val = None  # Non applicable
                            
                        # Cas 4: Autres méthodes sur intervalle infini
                        elif b == np.inf and meth_name in ['Simpson', 'Spline', 'Gauss-Legendre']:
                            val = meth_func(f, a, b, n)  # Transformations internes
                            
                        # Cas 5: Méthodes standard
                        else:
                            val = meth_func(f, a, b, n)
                        
                        temps_total += time.perf_counter() - start
                    
                    # ANALYSE DES RÉSULTATS
                    # =====================
                    if val is not None and not np.isnan(val) and not np.isinf(val):
                        # Calcul du temps moyen
                        temps_moyen = temps_total / repetitions
                        
                        # Calcul de l'erreur absolue
                        erreur_abs = abs(val - exacte)
                        
                        # Stockage des résultats
                        erreurs.append(erreur_abs)
                        temps_moy.append(temps_moyen)
                        n_vals.append(n)
                        
                        # Affichage détaillé
                        print(f"    n={n:3d}: I={val:.6e}, erreur={erreur_abs:.2e}, temps={temps_moyen:.2e}s")
                    else:
                        # Cas où la méthode n'est pas applicable
                        print(f"    n={n:3d}: Non applicable ou valeur non valide")
                        
                except Exception as e:
                    # Gestion des erreurs (singularités, etc.)
                    print(f"    n={n:3d}: ERREUR - {str(e)[:50]}")
                    continue
            
            # STOCKAGE FINAL POUR CETTE MÉTHODE
            if erreurs:
                resultats[nom][meth_name]['erreurs'] = erreurs
                resultats[nom][meth_name]['temps'] = temps_moy
                resultats[nom][meth_name]['n'] = n_vals
    
    return resultats, fonctions_config, methodes


# ============================================================================
# VISUALISATION DES RÉSULTATS
# ============================================================================
def tracer_graphiques(resultats, fonctions_config, methodes):
    """
    Fonction de visualisation qui génère 8 graphiques :
    - 4 graphiques d'erreurs (un par fonction)
    - 4 graphiques de temps (un par fonction)
    
    CARACTÉRISTIQUES DES GRAPHIQUES :
    - Échelles logarithmiques pour mieux voir les ordres de grandeur
    - Couleurs et marqueurs distincts pour chaque méthode
    - Grille pour faciliter la lecture
    - Légendes détaillées
    """
    
    # Configuration des couleurs (cohérentes entre les graphiques)
    couleurs = {
        'Gauss-Legendre': 'red',      # Rouge - Méthode classique
        'Gauss-Laguerre': 'blue',     # Bleu - Méthode exponentielle
        'Gauss-Chebyshev': 'green',   # Vert - Méthode oscillante
        'Simpson': 'orange',          # Orange - Méthode simple
        'Spline': 'purple'            # Violet - Méthode interpolatrice
    }
    
    # Configuration des marqueurs
    marqueurs = {
        'Gauss-Legendre': 'o',    # Cercle
        'Gauss-Laguerre': 's',    # Carré
        'Gauss-Chebyshev': '^',   # Triangle
        'Simpson': 'D',           # Diamant
        'Spline': 'v'             # Triangle inversé
    }
    
    # Création des figures (2x2 = 4 graphiques par figure)
    fig_erreurs, axes_erreurs = plt.subplots(2, 2, figsize=(14, 10))
    fig_temps, axes_temps = plt.subplots(2, 2, figsize=(14, 10))
    
    # Transformation en tableaux 1D pour faciliter l'indexation
    axes_erreurs = axes_erreurs.flatten()
    axes_temps = axes_temps.flatten()
    
    # ======================================================================
    # BOUCLE SUR LES FONCTIONS POUR LE TRACÉ
    # ======================================================================
    for idx, func_config in enumerate(fonctions_config):
        nom = func_config['nom']
        ax_err = axes_erreurs[idx]  # Axe pour les erreurs
        ax_tmp = axes_temps[idx]    # Axe pour les temps
        
        # CONFIGURATION DU GRAPHIQUE D'ERREURS
        # ====================================
        ax_err.set_title(f'Fonction {nom} - Erreurs', fontsize=12, fontweight='bold')
        ax_err.set_xlabel('n (nombre de points/subdivisions)', fontsize=10)
        ax_err.set_ylabel('log10(Erreur absolue)', fontsize=10)
        ax_err.grid(True, alpha=0.3, linestyle='--')  # Grille discrète
        
        # CONFIGURATION DU GRAPHIQUE DE TEMPS
        # ===================================
        ax_tmp.set_title(f'Fonction {nom} - Temps d\'exécution', fontsize=12, fontweight='bold')
        ax_tmp.set_xlabel('n (nombre de points/subdivisions)', fontsize=10)
        ax_tmp.set_ylabel('log10(Temps en secondes)', fontsize=10)
        ax_tmp.grid(True, alpha=0.3, linestyle='--')
        
        # ==================================================================
        # TRACÉ DE CHAQUE MÉTHODE
        # ==================================================================
        for meth_name in methodes.keys():
            if resultats[nom][meth_name]['n']:  # Vérifier si la méthode a été testée
                n_vals = resultats[nom][meth_name]['n']
                erreurs = resultats[nom][meth_name]['erreurs']
                temps = resultats[nom][meth_name]['temps']
                
                if erreurs and temps:
                    # Transformation logarithmique (éviter log(0))
                    erreurs_log = np.log10([max(e, 1e-16) for e in erreurs])
                    temps_log = np.log10([max(t, 1e-16) for t in temps])
                    
                    # TRACÉ DES ERREURS
                    ax_err.plot(n_vals, erreurs_log, 
                              color=couleurs[meth_name], 
                              marker=marqueurs[meth_name],
                              markersize=6,
                              linewidth=2,
                              label=meth_name)
                    
                    # TRACÉ DES TEMPS
                    ax_tmp.plot(n_vals, temps_log,
                              color=couleurs[meth_name],
                              marker=marqueurs[meth_name],
                              markersize=6,
                              linewidth=2,
                              label=meth_name)
        
        # Ajout des légendes
        ax_err.legend(fontsize=9, loc='best')
        ax_tmp.legend(fontsize=9, loc='best')
        
        # Échelle logarithmique en x pour mieux voir la progression
        ax_err.set_xscale('log')
        ax_tmp.set_xscale('log')
    
    # Optimisation de l'espacement
    fig_erreurs.tight_layout()
    fig_temps.tight_layout()
    
    # Sauvegarde des figures (pour le rapport)
    fig_erreurs.savefig('graphiques_erreurs.png', dpi=300, bbox_inches='tight')
    fig_temps.savefig('graphiques_temps.png', dpi=300, bbox_inches='tight')
    
    # Affichage des figures
    plt.show()
    
    return fig_erreurs, fig_temps


# ============================================================================
# ANALYSE DÉTAILLÉE DES RÉSULTATS
# ============================================================================
def analyser_resultats(resultats, fonctions_config):
    """
    Analyse approfondie des résultats avec calculs de :
    1. Meilleure méthode par fonction (précision)
    2. Méthode la plus rapide par fonction
    3. Taux de convergence de chaque méthode
    
    Le taux de convergence est calculé comme :
    r = log(erreur_n / erreur_{n/2}) / log(2)
    """
    
    print("\n" + "="*80)
    print("ANALYSE DÉTAILLÉE DES RÉSULTATS")
    print("="*80)
    
    for func_config in fonctions_config:
        nom = func_config['nom']
        print(f"\n{'='*60}")
        print(f"ANALYSE POUR LA FONCTION: {nom}")
        print(f"{'='*60}")
        
        # Initialisation des dictionnaires de comparaison
        meilleure_erreur = {}  # {méthode: erreur_min}
        meilleur_temps = {}    # {méthode: temps_min}
        
        # COLLECTE DES DONNÉES
        for meth_name in resultats[nom].keys():
            if resultats[nom][meth_name]['erreurs']:
                erreurs = resultats[nom][meth_name]['erreurs']
                temps = resultats[nom][meth_name]['temps']
                
                if erreurs:
                    erreur_min = min(erreurs)
                    meilleure_erreur[meth_name] = erreur_min
                
                if temps:
                    temps_min = min(temps)
                    meilleur_temps[meth_name] = temps_min
        
        # AFFICHAGE : MEILLEURE PRÉCISION
        if meilleure_erreur:
            print("\nMeilleure précision (erreur minimale):")
            # Tri par erreur croissante
            for meth, err in sorted(meilleure_erreur.items(), key=lambda x: x[1]):
                print(f"  {meth:20s}: {err:.2e}")
        
        # AFFICHAGE : MEILLEUR TEMPS
        if meilleur_temps:
            print("\nMeilleur temps d'exécution:")
            # Tri par temps croissant
            for meth, tps in sorted(meilleur_temps.items(), key=lambda x: x[1]):
                print(f"  {meth:20s}: {tps:.2e} s")
        
        # ANALYSE DE CONVERGENCE
        print("\nTaux de convergence (dernières valeurs):")
        for meth_name in resultats[nom].keys():
            n_vals = resultats[nom][meth_name]['n']
            erreurs = resultats[nom][meth_name]['erreurs']
            
            # Calcul du taux seulement si on a assez de points
            if len(n_vals) >= 2 and len(erreurs) >= 2:
                try:
                    # Calcul du taux entre les deux dernières valeurs
                    # r = log(erreur_n/erreur_{n/2}) / log(2)
                    if erreurs[-1] > 0 and erreurs[-2] > 0:
                        r = np.log(erreurs[-1]/erreurs[-2]) / np.log(n_vals[-2]/n_vals[-1])
                        print(f"  {meth_name:20s}: taux ≈ {r:.2f}")
                except:
                    pass  # Ignorer les erreurs de calcul


# ============================================================================
# GÉNÉRATION DU RAPPORT
# ============================================================================
def generer_rapport(resultats, fonctions_config):
    """
    Génère un rapport synthétique au format texte
    
    CONTENU DU RAPPORT :
    1. Introduction
    2. Description des fonctions tests
    3. Méthodologie
    4. Résultats synthétiques
    5. Conclusions
    6. Recommandations
    """
    
    rapport = """
RAPPORT DE COMPARAISON DES MÉTHODES D'INTÉGRATION NUMÉRIQUE
============================================================

1. INTRODUCTION
---------------
Ce rapport compare cinq méthodes d'intégration numérique appliquées à quatre
fonctions tests représentatives. Les méthodes étudiées sont:
1. Quadrature de Gauss-Legendre (classique)
2. Quadrature de Gauss-Laguerre
3. Quadrature de Gauss-Chebyshev
4. Méthode composite de Simpson
5. Intégration par spline cubique

2. FONCTIONS TESTS
------------------
"""
    
    # Description détaillée des fonctions
    for idx, func in enumerate(fonctions_config):
        rapport += f"""
2.{idx+1} Fonction {func['nom']}
   Intervalle: {func['intervalle']}
   Poids: {func['poids']}
   Valeur exacte: {func['exacte']:.10e}
   Caractéristiques: """
        
        if func['nom'] == 'Chebyshev':
            rapport += "Fonction oscillante avec poids 1/√(1-x²)"
        elif func['nom'] == 'Laguerre':
            rapport += "Décroissance exponentielle avec poids e^{-x}"
        elif func['nom'] == 'Combinee':
            rapport += "Combinaison des deux types de poids"
        else:
            rapport += "Fonction lisse sans singularité"
    
    rapport += """

3. MÉTHODOLOGIE
---------------
Pour chaque méthode et chaque fonction, on calcule:
- L'intégrale numérique pour n = 4, 8, 16, 32, 64, 128
- L'erreur absolue par rapport à la valeur exacte
- Le temps d'exécution moyen sur 5 répétitions

4. RÉSULTATS SYNTHÉTIQUES
-------------------------
"""
    
    # Tableau synthétique des meilleures méthodes
    rapport += "\nMéthode la plus précise par fonction:\n"
    rapport += "-" * 50 + "\n"
    
    for func in fonctions_config:
        nom = func['nom']
        meilleures = {}
        for meth in resultats[nom].keys():
            if resultats[nom][meth]['erreurs']:
                erreurs = resultats[nom][meth]['erreurs']
                if erreurs:
                    meilleures[meth] = min(erreurs)
        
        if meilleures:
            meilleure_meth = min(meilleures.items(), key=lambda x: x[1])
            rapport += f"{nom:15s}: {meilleure_meth[0]:20s} (erreur: {meilleure_meth[1]:.2e})\n"
    
    # Conclusions et recommandations
    rapport += """

5. CONCLUSIONS
--------------

5.1 Observations générales:
- Les méthodes de Gauss adaptées au poids (Chebyshev pour poids 1/√(1-x²), 
  Laguerre pour poids e^{-x}) sont les plus efficaces pour leurs fonctions cibles
- Gauss-Legendre est robuste et performante pour les fonctions lisses
- Simpson et Spline ont des performances correctes mais généralement inférieures
  aux méthodes de Gauss pour les fonctions tests considérées

5.2 Compromis précision/temps:
- Les méthodes de Gauss nécessitent moins de points pour atteindre une précision donnée
- Simpson est simple à implémenter mais nécessite plus de points pour une précision équivalente
- Spline cubique est coûteuse en temps pour un grand nombre de points

5.3 Recommandations:
- Utiliser Gauss-Chebyshev pour les intégrales avec poids 1/√(1-x²)
- Utiliser Gauss-Laguerre pour les intégrales sur [0,∞) avec poids e^{-x}
- Utiliser Gauss-Legendre pour les intégrales standards sur un intervalle fini
- Simpson est un bon choix pour des applications simples avec des fonctions régulières
- Spline peut être utile quand on a besoin à la fois de l'intégrale et d'une interpolation

6. FICHIERS GÉNÉRÉS
-------------------
- graphiques_erreurs.png : Graphiques des erreurs pour les 4 fonctions
- graphiques_temps.png : Graphiques des temps d'exécution pour les 4 fonctions
- rapport_integration.txt : Ce rapport
"""
    
    # Sauvegarde du rapport
    with open('rapport_integration.txt', 'w', encoding='utf-8') as f:
        f.write(rapport)
    
    print(rapport)
    print("\nRapport sauvegardé dans 'rapport_integration.txt'")


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    """
    SCRIPT PRINCIPAL - Point d'entrée du programme
    
    DÉROULEMENT :
    1. Affichage de l'en-tête
    2. Exécution des tests
    3. Génération des graphiques
    4. Analyse des résultats
    5. Génération du rapport
    6. Affichage des fichiers générés
    """
    
    print("DÉMARRAGE DE LA COMPARAISON DES MÉTHODES D'INTÉGRATION NUMÉRIQUE")
    print("="*70)
    
    # ÉTAPE 1 : Exécution des tests
    print("\nExécution des tests...")
    resultats, fonctions_config, methodes = tester_methodes()
    
    # ÉTAPE 2 : Génération des graphiques
    print("\nGénération des graphiques...")
    fig_erreurs, fig_temps = tracer_graphiques(resultats, fonctions_config, methodes)
    
    # ÉTAPE 3 : Analyse approfondie
    analyser_resultats(resultats, fonctions_config)
    
    # ÉTAPE 4 : Rapport final
    print("\nGénération du rapport...")
    generer_rapport(resultats, fonctions_config)
    
    # FIN DU PROGRAMME
    print("\n" + "="*70)
    print("PROGRAMME TERMINÉ AVEC SUCCÈS")
    print("="*70)
    print("\nFichiers générés:")
    print("1. graphiques_erreurs.png - Graphiques des erreurs")
    print("2. graphiques_temps.png - Graphiques des temps d'exécution")
    print("3. rapport_integration.txt - Rapport synthétique")
