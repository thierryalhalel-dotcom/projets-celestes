

import matplotlib.pyplot as plt
import numpy as np

# Fonction pour calculer l'equation du temps
def equation_du_temps(jour):
    B = (jour - 81) * 360 / 365
    E = 9.87 * np.sin(2 * np.radians(B)) - 7.53 * np.cos(np.radians(B)) - 1.5 * np.sin(np.radians(B))
    return E

# Generer les jours de l'annee 2024
jours = np.arange(1, 366)

# Calculer l'equation du temps pour chaque jour
equation_temps = [equation_du_temps(jour) for jour in jours]

# Tracer l'equation du temps
plt.figure(figsize=(10, 5))
plt.plot(jours, equation_temps, label="Equation du temps")
#plt.plot(jours, equation_temps, 'o')
plt.xlabel('Jour de l\'annee')
plt.ylabel('Equation du temps (minutes)')
plt.title('Equation du temps pour l\'annee 2024')
plt.legend()
plt.grid(True)
plt.show()