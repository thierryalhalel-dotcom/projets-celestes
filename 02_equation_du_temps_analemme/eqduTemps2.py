

import matplotlib.pyplot as plt
import numpy as np

# Fonction pour calculer l'equation du temps
def equation_du_temps(jour):
    B = (jour - 81) * 360 / 365
    E = 9.87 * np.sin(2 * np.radians(B)) - 7.53 * np.cos(np.radians(B)) - 1.5 * np.sin(np.radians(B))
    return E

# Generer les jours de l'annee
jours = np.arange(1, 366)

# Calculer l'equation du temps pour chaque annee
annees = [2024, 3024, 4024]
equations_temps = {annee: [equation_du_temps(jour) for jour in jours] for annee in annees}

# Tracer l'equation du temps pour chaque annee
plt.figure(figsize=(10, 5))
for annee in annees:
    plt.plot(jours, equations_temps[annee], label=f"Equation du temps {annee}")

plt.xlabel('Jour de l\'annee')
plt.ylabel('Equation du temps (minutes)')
plt.title('Equation du temps pour les annees 2024, 3024, 4024')
plt.legend()
plt.grid(True)
plt.show()