import pygame

WIDTH = 500
HEIGHT = 500
ROWS = 8
COLS = 8
SQUARE = WIDTH // COLS

# SOLO DEFINO ALGUNOS COLORES A USAR EN EL TABLERO

BLUE = (22, 83, 113)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (238, 242, 238)
RED = (255, 105, 97)

CROWN = pygame.Surface((44, 25))  # Crear una nueva superficie de tamaño (44, 25)
CROWN.fill((0, 0, 0))  # Llenar la superficie con color negro
