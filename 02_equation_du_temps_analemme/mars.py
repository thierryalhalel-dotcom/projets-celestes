


import matplotlib.pyplot as plt
import numpy as np

# Fonction pour calculer l'equation du temps sur Mars
def equation_du_temps_mars(jour):
    B = (jour - 81) * 360 / 687  # Mars a une annee de 687 jours
    E = 2.91 * np.sin(2 * np.radians(B)) - 2.11 * np.cos(np.radians(B)) - 1.0 * np.sin(np.radians(B))
    return E

# Fonction pour calculer la declinaison du soleil sur Mars
def declinaison_du_soleil_mars(jour):
    B = (jour - 81) * 360 / 687
    declinaison = 23.44 * np.sin(np.radians(B))  # Valeur approximative pour Mars
    return declinaison

# Generer les jours de l'annee martienne
jours = np.arange(1, 688)

# Calculer l'equation du temps et la declinaison pour chaque jour
equation_temps_mars = [equation_du_temps_mars(jour) for jour in jours]
declinaison_mars = [declinaison_du_soleil_mars(jour) for jour in jours]

# Tracer l'equation du temps en fonction des jours
plt.figure(figsize=(10, 5))
plt.plot(jours, equation_temps_mars, label="Equation du temps sur Mars")
plt.xlabel('Jour de l\'annee martienne')
plt.ylabel('Equation du temps (minutes)')
plt.title('Equation du temps sur Mars pour l\'annee 2024')
plt.legend()
plt.grid(True)
plt.show()

# Tracer l'equation du temps en fonction de la declinaison du soleil
plt.figure(figsize=(10, 5))
plt.plot(declinaison_mars, equation_temps_mars, label="Equation du temps vs Declinaison")
plt.xlabel('Declinaison du soleil (degres)')
plt.ylabel('Equation du temps (minutes)')
plt.title('Equation du temps en fonction de la declinaison du soleil sur Mars')
plt.legend()
plt.grid(True)
plt.show()