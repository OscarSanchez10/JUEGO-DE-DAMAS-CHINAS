import pygame
from constants import *


class Piece:
    OUT = 2
    PADDING = 12

    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color
        self.king = False
        self.x = 0
        self.y = 0
        self.calc_position()

    def calc_position(self):
        # Centramos la pieza
        self.x = SQUARE*self.col + SQUARE//2
        self.y = SQUARE*self.row + SQUARE//2

    def make_king(self):
        self.king = True

    def draw(self, window):
        radius = SQUARE//2 - self.PADDING
        pygame.draw.circle(window, WHITE, (self.x, self.y), radius+self.OUT)
        pygame.draw.circle(window, self.color, (self.x, self.y), radius)
        if self.king:
            window.blit(CROWN, (self.x - CROWN.get_width() //
                        2, self.y - CROWN.get_height()//2))

    def move(self, row, col):
        self.row = row
        self.col = col
        self.calc_position()

    def is_king(self):
        return self.king

    def __repr__(self):
        # Devolvemos el color de la pieza en forma de cadena
        return str(self.color)
