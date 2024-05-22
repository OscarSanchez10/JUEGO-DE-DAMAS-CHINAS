import pygame
import os
import random
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from constants import *
from piece import *


def minimax(position, depth, max_player, game, population_size, num_generations):
    # la profundidad es 0 o si hay un ganador, se devuelve la evaluación de la posición actual.
    if depth == 0 or position.winner() != None:
        return position.evaluate(), position
    # Si el jugador actual es el jugador máximo, se busca el mejor movimiento para el jugador.
    if max_player:
        maxEval = float("-inf")
        best_move = None

        best_individual = genetic_algorithm(
            position, WHITE, game, population_size, num_generations
        )

        # Graficar la curva de aprendizaje
        #plot_learning_curve('best_fitness_scores.txt')

        for move in get_all_moves(position, WHITE, game):
            # Se obtiene la evaluación del movimiento actual.
            evaluation = minimax(
                move, depth - 1, False, game, population_size, num_generations
            )[0]
            # Se actualiza el valor máximo y se guarda el movimiento actual si la evaluación es igual.
            maxEval = max(maxEval, evaluation)
            if maxEval == evaluation:
                best_move = move
                best_individual_eval = evaluate_board(best_individual, WHITE)

        print("\nMaxEval:", maxEval)
        print("Best Move:", best_move)
        print("Best Individual Eval:", best_individual_eval)

        return maxEval, best_move
    # Si el jugador actual no es el jugador máximo, se busca el mejor movimiento para el oponente.
    else:
        minEval = float("inf")
        best_move = None
        best_individual = None
        best_individual = genetic_algorithm(
            position, WHITE, game, population_size, num_generations
        )
        for move in get_all_moves(position, BLACK, game):
            evaluation = minimax(
                move, depth - 1, True, game, population_size, num_generations
            )[0]
            # Se actualiza el valor mínimo y se guarda el movimiento actual si la evaluación es igual.
            minEval = min(minEval, evaluation)
            if minEval == evaluation:
                best_move = move
                best_individual_eval = evaluate_board(best_individual, BLACK)

        print("\nMinEval:", minEval)
        print("Best Move:", best_move)
        print("Best Individual Eval:", best_individual_eval, "\n")

        return minEval, best_move


def save_population(population, filename):
    with open(filename, "wb") as file:
        pickle.dump(population, file)


def load_population(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def simulate_move(piece, move, board, game, skip):
    board.move(piece, move[0], move[1])
    # Si se saltó una pieza, se elimina del tablero.
    if skip:
        board.remove(skip)

    return board


def get_all_moves(board, color, game):
    moves = []

    for piece in board.get_all_pieces(color):
        valid_moves = board.get_valid_moves(piece)
        for move, skip in valid_moves.items():
            #draw_moves(game, board, piece)
            # Se crea una copia del tablero y de la pieza para simular el movimiento.
            temp_board = deepcopy(board)
            temp_piece = temp_board.get_piece(piece.row, piece.col)
            new_board = simulate_move(temp_piece, move, temp_board, game, skip)
            moves.append(new_board)

    return moves


def evaluate_board(board, color):
    """
    Evalúa la calidad del estado actual del tablero.
    Cuanto mayor sea la puntuación, mejor será el estado para el jugador.
    """
    score = 0

    # Contar el número de piezas en el tablero
    player_pieces = board.get_all_pieces(color)
    opponent_color = BLACK if color == WHITE else WHITE
    opponent_pieces = board.get_all_pieces(opponent_color)
    num_player_pieces = len(player_pieces)
    num_opponent_pieces = len(opponent_pieces)

    # Heurística de conteo de piezas
    score += num_player_pieces - num_opponent_pieces

    # Heurística de posición de piezas
    for piece in player_pieces:
        # El peso de la pieza es mayor en el lado del oponente del tablero
        row_weight = (7 - piece.row) if color == WHITE else piece.row
        score += row_weight

        # El peso de la pieza también es mayor para las piezas más cercanas al lado del oponente
        distance_weight = abs(piece.row - 7) if color == WHITE else abs(piece.row)
        score += distance_weight

        # Las piezas rey tienen mayor peso ya que pueden moverse en todas las direcciones
        if piece.is_king():
            score += 0.5

        # Bonificación defensiva para las piezas en la fila trasera
        if piece.row == (7 if color == BLACK else 0):
            score += 0.25

    # Comprobar si el jugador ha ganado el juego
    if board.winner() == color:
        score += 100

    # Comprobar si el jugador ha perdido el juego
    elif board.winner() == opponent_color:
        score -= 100

    return score


def generate_population(board, color, game, size):
    population = []
    for i in range(size):
        individual = []
        temp_board = deepcopy(board)
        for j in range(10):
            moves = get_all_moves(temp_board, color, game)
            if moves:
                move = random.choice(moves)
                individual.append(move)
                temp_board = move
            else:
                break
        fitness = evaluate_board(temp_board, color)
        population.append({"moves": individual, "fitness": fitness})
    return population


def tournament_selection(population, num_parents):
    parents = []
    for i in range(num_parents):
        # Seleccionar dos índices aleatorios del conjunto de población
        indices = random.sample(range(len(population)), 2)
        print("\nSeleccion por torneo:", indices)
        # Extraer los individuos correspondientes a esos índices
        tournament = [population[indices[0]], population[indices[1]]]
        print("\nIndividuos", tournament)

        # Comparar el desempeño de cada individuo usando la aptitud almacenada
        if tournament[0]["fitness"] >= tournament[1]["fitness"]:
            parents.append(tournament[0])
        else:
            parents.append(tournament[1])
    return parents


def crossover(parent1, parent2):
    """
    Función de cruce que combina dos soluciones de padres para producir una nueva descendencia.
    En el juego de damas, intercambiamos aleatoriamente algunos movimientos entre los dos padres para crear una nueva solución.
    """
    # Seleccionamos un punto aleatorio para dividir las soluciones de los padres
    split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    print("\nCorte", split_point)

    # Combinamos las soluciones de los dos padres intercambiando algunos movimientos
    offspring_moves = parent1["moves"][:split_point] + parent2["moves"][split_point:]

    print("\nHijo:", offspring_moves)

    moves_count = len(offspring_moves)
    for i in range(moves_count):
        offspring_fitness = evaluate_board(offspring_moves[i], WHITE)
    # Creamos un nuevo diccionario de descendencia con los movimientos combinados y la puntuación de aptitud
    offspring = {"moves": offspring_moves, "fitness": offspring_fitness}

    return offspring


def mutate(child, color):
    """
    Cambia aleatoriamente una parte de una solución (en este caso, una secuencia de movimientos)
    para producir una variación. Por ejemplo, se podría cambiar aleatoriamente un movimiento en la secuencia.
    """
    moves = child["moves"]

    if not moves:
        return child

    # Seleccionar un índice aleatorio de movimientos.
    index = random.randint(0, len(moves) - 1)

    # Obtener el tablero en el índice seleccionado.
    board = moves[index]

    # Obtener una pieza aleatoria del tablero.
    piece = random.choice(board.get_all_pieces(color))

    # Obtener todos los movimientos válidos de la pieza.
    valid_moves = board.get_valid_moves(piece)

    if not valid_moves:
        return child

    # Seleccionar un movimiento aleatorio.
    move, skip = random.choice(list(valid_moves.items()))

    # Simular el movimiento en una copia del tablero para obtener una nueva configuración.
    temp_board = deepcopy(board)
    temp_piece = temp_board.get_piece(piece.row, piece.col)
    new_board = simulate_move(temp_piece, move, temp_board, None, skip)

    # Reemplazar el tablero en el índice seleccionado con el tablero mutado.
    moves[index] = new_board

    # Actualizar la aptitud del hijo.
    child["fitness"] = evaluate_board(new_board, color)

    return child


def replacement(population, offspring):
    # Combinar la población actual y los descendientes
    combined_population = population + offspring

    # Ordenar la población combinada por orden descendente de aptitud
    sorted_population = sorted(
        combined_population, key=lambda x: x["fitness"], reverse=True
    )

    # Seleccionar los individuos menos aptos de la población actual
    worst_individuals = sorted_population[len(population) :]

    # Reemplazar los individuos menos aptos con los descendientes más aptos
    for i in range(len(worst_individuals)):
        new_individual = dict(worst_individuals[i])
        new_individual["fitness"] = sorted_population[i]["fitness"]
        worst_individuals[i] = new_individual

    return sorted_population[: len(population)]


def select_board(population, generation):
    index = generation % len(population)
    selected_board = population[index]["moves"][-1]
    return selected_board


def save_best_fitness_scores(filename, best_fitness_scores):
    with open(filename, "a") as file:
        for score in best_fitness_scores:
            file.write(str(score) + "\n")


def genetic_algorithm(board, color, game, population_size, num_generations):
    best_fitness_scores = []

    filename = "population.pkl"

    # Verificar si existe el archivo de población guardado
    if os.path.exists(filename):
        # Cargar la población del archivo
        population = load_population(filename)
        print("\nSe cargo Correctamente", population)
    else:
        # Generar una nueva población
        population = generate_population(board, color, game, population_size)

    # Imprime el contenido de la población
    for index, individual in enumerate(population, start=1):
        print(f"\nIndividual {index}: {individual}\n")

    for generation in range(num_generations):
        print(f"\nGeneración {generation + 1}:")
        best_fitness_scores.append(
            max(individual["fitness"] for individual in population)
        )
        # Seleccionar un tablero en función del número de generación
        selected_board = select_board(population, generation)
        board_evaluation = evaluate_board(selected_board, color)
        print("\nEvaluación del tablero seleccionado:", board_evaluation)

        # Selección de padres
        parents = tournament_selection(population, 2)

        # Generación de descendencia mediante cruce y mutación
        offspring = []
        for j in range(0, len(parents), 2):
            parent1 = parents[j]
            print("\nPadre 1", parent1)
            parent2 = parents[j + 1]
            print("\nPadre 2", parent2)
            child = crossover(parent1, parent2)
            print("\nChild", child)
            mutated_child = mutate(child, color)
            print("\nMutate Child", mutated_child)
            offspring.append(mutated_child)

        # Reemplazo de los individuos menos aptos por los más aptos de la descendencia
        population = replacement(population, offspring)

        save_population(population, "population.pkl")

    # Devolver el mejor individuo de la última generación
    best_individual = max(population, key=lambda x: x["fitness"])
    best_board = best_individual["moves"][-1]

    save_best_fitness_scores("best_fitness_scores.txt", best_fitness_scores)

    return best_board


def plot_learning_curve(filename):
    all_best_fitness_scores = []
    current_scores = []

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line == "---":
                all_best_fitness_scores.append(current_scores)
                current_scores = []
            else:
                current_scores.append(float(line))

        if current_scores:
            all_best_fitness_scores.append(current_scores)

    for best_fitness_scores in all_best_fitness_scores:
        plt.plot(best_fitness_scores)

    plt.xlabel("Generations")
    plt.ylabel("Best Fitness Score")
    plt.title("Genetic Algorithm Learning Curve")
    plt.show()


def draw_moves(game, board, piece):
    valid_moves = board.get_valid_moves(piece)
    board.draw(game.window)
    pygame.draw.circle(game.window, (0, 255, 0), (piece.x, piece.y), 30, 5)
    game.draw_valid_moves(valid_moves.keys())
    pygame.display.update()
    pygame.time.delay(500)
