import matplotlib.pyplot as plt
from datetime import datetime

# Dates des equinoxes et solstices pour 2024
dates = {
    "Printemps": datetime(2024, 3, 20),
    "ete": datetime(2024, 6, 20),
    "Automne": datetime(2024, 9, 22),
    "Hiver": datetime(2024, 12, 21)
}

# Calculer la duree de chaque saison
durees = {
    "Printemps": (dates["ete"] - dates["Printemps"]).days,
    "ete": (dates["Automne"] - dates["ete"]).days,
    "Automne": (dates["Hiver"] - dates["Automne"]).days,
    "Hiver": (datetime(2025, 3, 20) - dates["Hiver"]).days  # Jusqu'a l'equinoxe de printemps 2025
}

# Afficher les durees des saisons
for saison, duree in durees.items():
    print(f"{saison}: {duree} jours")

# Representer graphiquement la duree des saisons
plt.figure(figsize=(10, 5))
plt.bar(durees.keys(), durees.values(), color=['green', 'yellow', 'orange', 'blue'])
plt.xlabel('Saison')
plt.ylabel('Duree (jours)')
plt.title('Duree des saisons astronomiques en 2024')
plt.show()