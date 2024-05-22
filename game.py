import pygame
from board import *
from constants import *


class Game:
    def __init__(self, window=None):
        self._init()
        self.window = window

    def update(self):
        if self.window:  # Comprueba si la ventana existe antes de intentar usarla
            self.board.draw(self.window)
            self.draw_valid_moves(self.valid_moves)
            pygame.display.update()

    def _init(self):
        self.selected = None
        self.board = Board()
        self.turn = BLACK
        self.valid_moves = {}

    def winner(self):
        return self.board.winner()

    def reset(self):
        self._init()

    def select(self, row, col):
        if self.selected:
            result = self._move(row, col)
            if not result:
                self.selected = None
                self.select(row, col)

        piece = self.board.get_piece(row, col)
        if piece != 0 and piece.color == self.turn:
            self.selected = piece
            self.valid_moves = self.board.get_valid_moves(piece)
            return True

        return False

    def _move(self, row, col):
        piece = self.board.get_piece(row, col)
        if self.selected and piece == 0 and (row, col) in self.valid_moves:
            self.board.move(self.selected, row, col)
            skipped = self.valid_moves[(row, col)]
            if skipped:
                self.board.remove(skipped)
            self.change_turn()
        else:
            return False

        return True

    def network_move(self, move):
        self.board.move(move[0], move[1], move[2], move[3])

    def draw_valid_moves(self, moves):
        if self.window:  # Comprueba si la ventana existe antes de intentar usarla
            for move in moves:
                row, col = move
                pygame.draw.circle(self.window, RED, (col * SQUARE + SQUARE // 2, row * SQUARE + SQUARE // 2), 30)

    def change_turn(self):
        self.valid_moves = {}
        if self.turn == BLACK:
            self.turn = WHITE
        else:
            self.turn = BLACK

    def getBoard(self):
        return self.board

    def ai_move(self, board):
        self.board = board
        self.change_turn()
