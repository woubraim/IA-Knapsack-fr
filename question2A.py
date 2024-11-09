

# Résolution du problème du sac à dos ou KP (Knapsack Problem) à l'aide d'algorithme génétique
# import de la librairie

import numpy as np  # utilisation des calculs matriciels
import pandas as pd  # générer les fichiers csv
import random as rd  # génération de nombres aléatoires
from random import randint  # génération des nombres aléatoires
import matplotlib.pyplot as plt
import time

print("Approche numéro 1")

# Lecture de l'instance sauvegardée
instance = pd.read_csv("./instancesQ2/instance1.csv")
#instance = pd.read_csv("./instancesQ2/instance2.csv")
#instance = pd.read_csv("./instancesQ2/instance3.csv")
#instance = pd.read_csv("./instancesQ2/instance4.csv")

ID_objets = instance["ID"].to_numpy()
poids = instance["Poids_objets"].to_numpy()
valeur = instance["Valeur_objets"].to_numpy()

# Données du problème générées aléatoirement
nombre_objets = ID_objets.shape[0]  # Le nombre d'objets
capacite_max = 30  # La capacité du sac

# paramètres de l'algorithme génétique
nbr_generations = 150  # nombre de générations

# affichage des objets: Une instance aléatoire du problème Knapsack
print('La liste des objet est la suivante :')
print('ID_objet   Poids   Valeur')
for i in range(ID_objets.shape[0]):
    print(f'{ID_objets[i]} \t {poids[i]} \t {valeur[i]}')
print()

# Créer la population initiale
solutions_par_pop = 8  # la taille de la population
pop_size = (solutions_par_pop, ID_objets.shape[0])
population_initiale = np.random.randint(2, size=pop_size)
population_initiale = population_initiale.astype(int)

print(f'Taille de la population: {pop_size}')
print(f'Population Initiale: \n{population_initiale}')


def cal_fitness(poids, valeur, population, capacite):
    fitness = np.empty(population.shape[0])

    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * valeur)
        S2 = np.sum(population[i] * poids)

        if S2 <= capacite:
            fitness[i] = S1
        else:
            fitness[i] = capacite - S2

    return fitness.astype(int)


def selection(fitness, nbr_parents, population):
    fitness = list(fitness)
    parents = np.empty((nbr_parents, population.shape[1]))

    for i in range(nbr_parents):
        indice_max_fitness = np.where(fitness == np.max(fitness))
        parents[i, :] = population[indice_max_fitness[0][0], :]
        fitness[indice_max_fitness[0][0]] = -999999

    return parents


def croisement(parents, nbr_enfants):
    enfants = np.empty((nbr_enfants, parents.shape[1]))
    point_de_croisement = int(parents.shape[1] / 2)  # croisement au milieu
    taux_de_croisement = 0.8
    i = 0

    while (i < nbr_enfants):  # parents.shape[0]
        indice_parent1 = i % parents.shape[0]
        indice_parent2 = (i + 1) % parents.shape[0]
        x = rd.random()
        if x > taux_de_croisement:  # probabilité de parents stériles
            continue
        indice_parent1 = i % parents.shape[0]
        indice_parent2 = (i + 1) % parents.shape[0]
        enfants[i, 0:point_de_croisement] = parents[indice_parent1, 0:point_de_croisement]
        enfants[i, point_de_croisement:] = parents[indice_parent2, point_de_croisement:]
        i += 1

    return enfants


# La mutation consiste à inverser le bit
def mutation(enfants):
    mutants = np.empty((enfants.shape))
    taux_mutation = 0.5
    for i in range(mutants.shape[0]):
        random_valeur = rd.random()
        mutants[i, :] = enfants[i, :]
        if random_valeur > taux_mutation:
            continue
        int_random_valeur = randint(0, enfants.shape[1] - 1)  # choisir aléatoirement le bit à inverser
        if mutants[i, int_random_valeur] == 0:
            mutants[i, int_random_valeur] = 1
        else:
            mutants[i, int_random_valeur] = 0
    return mutants


def optimize(poids, valeur, population, pop_size, nbr_generations, capacite):
    sol_opt, historique_fitness = [], []
    nbr_parents = pop_size[0] // 2
    nbr_enfants = pop_size[0] - nbr_parents
    for _ in range(nbr_generations):
        fitness = cal_fitness(poids, valeur, population, capacite)
        # faire le traitement ici

        historique_fitness.append(fitness)
        parents = selection(fitness, nbr_parents, population)
        enfants = croisement(parents, nbr_enfants)
        mutants = mutation(enfants)

        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    print(f'Voici la dernière génération de la population: \n{population}\n')
    fitness_derniere_generation = cal_fitness(poids, valeur, population, capacite)
    print(f'Fitness de la dernière génération: \n{fitness_derniere_generation}\n')
    max_fitness = np.where(fitness_derniere_generation == np.max(fitness_derniere_generation))
    sol_opt.append(population[max_fitness[0][0], :])

    return sol_opt, historique_fitness


start = time.time()
# lancement de l'algorithme génétique
sol_opt, historique_fitness = optimize(poids, valeur, population_initiale, pop_size, nbr_generations, capacite_max)

end = time.time()

print('Temps dexecution: ', end - start)

# affichage du résultat
print('La solution optimale est:')
print('objets n°', [i for i, j in enumerate(sol_opt[0]) if j != 0])

print(f'Avec une valeur de {np.amax(historique_fitness)} € et un poids de {np.sum(sol_opt * poids)} kg')
print('Les objets qui maximisent la valeur contenue dans le sac sans le déchirer :')
objets_selectionnes = ID_objets * sol_opt
for i in range(objets_selectionnes.shape[1]):
    if ((sol_opt[0][i] == 1)):
        print(f'{objets_selectionnes[0][i]}')

historique_fitness_moyenne = [np.mean(fitness) for fitness in historique_fitness]
historique_fitness_max = [np.max(fitness) for fitness in historique_fitness]
plt.plot(list(range(nbr_generations)), historique_fitness_moyenne, label='Valeurs moyennes')
plt.plot(list(range(nbr_generations)), historique_fitness_max, label='Valeur maximale')
plt.legend()
plt.title('Evolution de la Fitness à travers les générations en Euros')
plt.xlabel('Générations')
plt.ylabel('Fitness')
plt.show()
