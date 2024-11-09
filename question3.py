
# Appliquer l'algorithme à la gestion du portefeuille
# import de la librairie

import numpy as np  # utilisation des calculs matriciels
import pandas as pd  # générer les fichiers csv
import random as rd  # génération de nombres aléatoires
from random import randint  # génération des nombres aléatoires
import matplotlib.pyplot as plt
import time

# Données du problème générées aléatoirement
nombre_titres = 20  # Le nombre de titres
budget_total = 1000  # Le budget total
LIMITE = 0.2  # limite de 20%
critere_risque = int(LIMITE * budget_total)  # critere de minimisation du risque: 20% du budget total

# paramètres de l'algorithme génétique
nbr_generations = 1000  # nombre de générations

titres = np.arange(0, nombre_titres)  # ID des titres à acheter de 1 à nombre_titres

actions = np.random.randint(50, 350, size=nombre_titres)  # Valeurs des titres générées aléatoirement entre 50€ et 300€

recap = pd.DataFrame({"ID": titres,
                      "Valeur_actions": actions,
                      })

recap.to_csv("./instancesQ4/instance4.csv", mode='a', index=False, header=True)

# affichage des objets: Une instance aléatoire du problème Knapsack
print('La liste des titres avec leur action est la suivante :')
print('ID_objet   actions')
for i in range(titres.shape[0]):
    print(f'{titres[i]} \t {actions[i]}')
print()


def verif_contrainte(population, actions):
    for i in range(population.shape[0]):
        # print(f'Solution à lentree : \n{population[i]}')

        # vecteur contenant l'indice des actions achetées plus d'une fois
        plus_un = [nbr_action for nbr_action in range(population.shape[1]) if population[i][nbr_action] > 1]

        if (len(plus_un) >= 1):
            if (len(plus_un) > 1):
                r = rd.choice(plus_un)  # choisir un indice au hasard
                population[i][r] = 1
                plus_un.remove(r)
            # diminuer le nombre d'actions achetées tant que ça depasse 20% du budget ou atteint 1 action (pour pas changé la solution)
            while (population[i][plus_un[0]] * actions[plus_un[0]] > critere_risque
                   and
                   population[i][plus_un[0]] != 1):
                population[i][plus_un[0]] -= 1

        # print(f'Solution à la sortie : \n{population[i]}')


def correction(population, actions, budget, critere_risque):
    for i in range(population.shape[0]):
        new = np.zeros(population.shape[1])
        if (np.sum(population[i] * actions) > budget):
            # recuperer l'indice de chaque 1 dans la solution
            ones = [one for one in range(population.shape[1]) if population[i][one] >= 1]
            rd.shuffle(ones)
            count = 0;
            j = 0

            while (j < len(ones)):
                if (population[i][ones[j]] * actions[ones[j]] > critere_risque):
                    population[i][ones[j]] -= 1
                    continue
                else:
                    count += population[i][ones[j]] * actions[ones[j]]
                    new[ones[j]] = population[i][ones[j]]
                if (count >= budget):
                    break
                j += 1
        population[i] = new


def verif_depassement(population, actions, budget):
    for i in range(population.shape[0]):
        if (np.sum(population[i] * actions) > budget):
            # recuperer l'indice de chaque 1 dans la solution
            ones = [one for one in range(population.shape[1]) if population[i][one] >= 1]
            while (np.sum(population[i] * actions) > budget):
                r = rd.choice(ones)  # choisir l'indice d'un 1 au hasard pour le mettre à 0
                if (population[i][r] != 0):
                    population[i][r] -= 1


# Créer la population initiale
solutions_par_pop = 10  # la taille de la population
pop_size = (solutions_par_pop, titres.shape[0])
population_initiale = np.random.randint(2, size=pop_size)
population_initiale = population_initiale.astype(int)

# L’achat de plusieurs actions ne concerne qu’un seul titre -> choisir un titre au hasard
for i in range(population_initiale.shape[0]):
    poids = np.random.randint(1, 20)  # Poids entre 1 et 10 à affecter à un titre càd acheter max 10 action / titre
    r = rd.choice(titres)
    population_initiale[i][r] = poids

print(f'Taille de la population: {pop_size}')
print(f'Population Initiale: \n{population_initiale}')

# verif_contrainte(population_initiale, actions)
# print(f'Population apres verification: \n{population_initiale}')
correction(population_initiale, actions, budget_total, critere_risque)
print(f'Population apres verif depassement: \n{population_initiale}')


def cal_fitness(actions, population, budget, critere_risque):
    fitness = np.empty(population.shape[0])

    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * actions)

        if S1 <= budget:
            fitness[i] = S1
            for j in range(population.shape[1]):
                # Si le nombre d'actions d'un titre depasse le critère risque -> pénaliser la fitness proportionnelle au depassement
                if (population[i][j] * actions[j] > critere_risque):
                    fitness[i] -= population[i][j] * actions[j]
        else:
            fitness[i] = 0

    return fitness.astype(int)


print(
    f'Fitness de la population initiale: \n{cal_fitness(actions, population_initiale, budget_total, critere_risque).sort()}\n')


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

    while (i < nbr_enfants):
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

    return enfants.astype(int)


# La mutation consiste à inverser le bit
def mutation(enfants):
    mutants = np.empty((enfants.shape))
    taux_mutation = 0.5
    for i in range(mutants.shape[0]):
        random_valeur = rd.random()
        mutants[i, :] = enfants[i, :]
        if random_valeur > taux_mutation:
            continue
        # choisir aléatoirement les bit à échanger
        int_random_valeur1 = randint(0, enfants.shape[1] - 1)
        int_random_valeur2 = randint(0, enfants.shape[1] - 1)
        # interchanger les valeur de deux titres au hasard
        # mutants[i,int_random_valeur1], mutants[i,int_random_valeur2] = mutants[i,int_random_valeur2], mutants[i,int_random_valeur1]

        # choisir aléatoirement le bit à changer
        mutants[i, int_random_valeur1] += 1  # maybe mutate multiple genes
        """ 
        if mutants[i,int_random_valeur] > 0:
            mutants[i,int_random_valeur] -= 1
        else:
            mutants[i,int_random_valeur] += 1
         """

    return mutants.astype(int)


def optimize(actions, population, pop_size, nbr_generations, budget, critere_risque):
    sol_opt, historique_fitness = [], []
    nbr_parents = pop_size[0] // 2
    nbr_enfants = pop_size[0] - nbr_parents
    for _ in range(nbr_generations):
        fitness = cal_fitness(actions, population, budget, critere_risque)
        historique_fitness.append(fitness)
        parents = selection(fitness, nbr_parents, population)

        enfants = croisement(parents, nbr_enfants)
        # verif_contrainte(enfants, actions)
        # verif_depassement(enfants, actions, budget)

        mutants = mutation(enfants)
        # verif_contrainte(mutants, actions)
        # verif_depassement(mutants, actions, budget)
        correction(mutants, actions, budget, critere_risque)

        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    print(f'Voici la dernière génération de la population: \n{population}\n')
    fitness_derniere_generation = cal_fitness(actions, population, budget, critere_risque)
    print(f'Fitness de la dernière génération: \n{fitness_derniere_generation}\n')
    max_fitness = np.where(fitness_derniere_generation == np.max(fitness_derniere_generation))
    sol_opt.append(population[max_fitness[0][0], :])

    return sol_opt, historique_fitness


start = time.time()
# lancement de l'algorithme génétique
sol_opt, historique_fitness = optimize(actions, population_initiale, pop_size, nbr_generations, budget_total,
                                       critere_risque)

end = time.time()

# affichage du résultat
print('La solution optimale est:')
print('Titre n°', [(i, j) for i, j in enumerate(sol_opt[0]) if j != 0])

print(f'Avec une valeur de {np.amax(historique_fitness)} € sur un budget total de {budget_total} €')
print('Les Titres qui maximisent le budget total de notre portefeuille :')
objets_selectionnes = titres * sol_opt
for i in range(objets_selectionnes.shape[1]):
    if ((sol_opt[0][i] >= 1)):
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