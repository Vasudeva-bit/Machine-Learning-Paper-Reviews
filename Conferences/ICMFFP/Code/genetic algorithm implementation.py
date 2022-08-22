import random
def objective(x, y, z):
    return 6*x**3 + 9*y**2 + 90*z - 25

def fitness(x, y, z):
    score = objective(x, y, z)
    
    if score == 0:
        return 99999
    else:
        return abs(1/score)

# generate the solutions
solutions = []
for s in range(1000):
    solutions.append((random.uniform(0, 10000),
                      random.uniform(0, 10000),
                      random.uniform(0, 10000)))

for i in range(10000):
    rankedSolutions = []
    for s in solutions:
        rankedSolutions.append((fitness(s[0], s[1], s[2]), s))
    rankedSolutions.sort()
    rankedSolutions.reverse()
    print(f"Genetic Algorithm {i} best solution ")
    print(rankedSolutions[0])

    if rankedSolutions[0][0] > 999:
        break

    # combine the best solutions to further find the better solutions based on the current best
    bestSolutions = rankedSolutions[:100]
    elements = []
    for s in bestSolutions:
        elements.append(s[1][0])
        elements.append(s[1][1])
        elements.append(s[1][2])
    # new generation from the existing best solutions
    newSolutions = []
    for _ in range(1000):
        e1 = random.choice(elements) * random.uniform(0.99, 1.01)
        e2 = random.choice(elements) * random.uniform(0.99, 1.01)
        e3 = random.choice(elements) * random.uniform(0.99, 1.01)

        newSolutions.append((e1, e2, e3))
    solutions = newSolutions