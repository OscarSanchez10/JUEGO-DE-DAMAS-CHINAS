import pygame
from constants import *
from piece import Piece


class Board:
    def __init__(self):
        self.board = []
        # Iniciales fichas de cada jugador
        self.black_left = self.white_left = 12
        self.black_kings = self.white_kings = 0
        self.create_board()

    # Metodo para Dibujar el tablero
    def draw_square(self, window):
        window.fill(RED)
        for row in range(ROWS):
            for col in range(row % 2, ROWS, 2):
                pygame.draw.rect(
                    window, GRAY, (row * SQUARE, col * SQUARE, SQUARE, SQUARE)
                )

    # Evaluar cuantas fichas se tienen
    def evaluate(self):
        return (
            self.white_left
            - self.black_left
            + (self.white_kings * 0.5 - self.black_kings * 0.5)
        )

    # retorna todas las fichas
    def get_all_pieces(self, color):
        pieces = []
        for row in self.board:
            for piece in row:
                if piece != 0 and piece.color == color:
                    pieces.append(piece)
        return pieces

    def move(self, piece, row, col):
        self.board[piece.row][piece.col], self.board[row][col] = (
            self.board[row][col],
            self.board[piece.row][piece.col],
        )
        piece.move(row, col)

        if row == ROWS - 1 or row == 0:
            piece.make_king()
            if piece.color == WHITE:
                self.white_kings += 1
            else:
                self.black_kings += 1

    def get_piece(self, row, col):
        return self.board[row][col]

    # Ubicacion de las fichas
    def create_board(self):
        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                if col % 2 == ((row + 1) % 2):
                    if row < 3:
                        self.board[row].append(Piece(row, col, WHITE))
                    elif row > 4:
                        self.board[row].append(Piece(row, col, BLACK))
                    else:
                        self.board[row].append(0)
                else:
                    self.board[row].append(0)

    # Metodo Para dibujar el tablero y fichas
    def draw(self, window):
        self.draw_square(window)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece != 0:
                    piece.draw(window)

    def remove(self, pieces):
        for piece in pieces:
            self.board[piece.row][piece.col] = 0
            if piece != 0:
                if piece.color == BLACK:
                    self.black_left -= 1
                else:
                    self.white_left -= 1

    def winner(self):
        if self.black_left <= 0:
            return WHITE
        elif self.white_left <= 0:
            return BLACK

        return None

    def get_valid_moves(self, piece):
        moves = {}
        left = piece.col - 1
        right = piece.col + 1
        row = piece.row

        # Condicion para la direccion de las fichas

        if piece.color == BLACK or piece.king:
            # Si la pieza es negra o un rey negro, entonces se pueden mover hacia arriba
            moves.update(
                self._traverse_left(row - 1, max(row - 3, -1), -1, piece.color, left)
            )
            moves.update(
                self._traverse_right(row - 1, max(row - 3, -1), -1, piece.color, right)
            )

        if piece.color == WHITE or piece.king:
            # Si la pieza es blanca o un rey blanco, entonces se pueden mover hacia abajo
            moves.update(
                self._traverse_left(row + 1, min(row + 3, ROWS), 1, piece.color, left)
            )
            moves.update(
                self._traverse_right(row + 1, min(row + 3, ROWS), 1, piece.color, right)
            )

        return moves

    def _traverse_left(self, start, stop, step, color, left, skipped=[]):
        # Función recursiva para buscar movimientos posibles hacia la izquierda
        moves = {}
        last = []
        for r in range(start, stop, step):
            if left < 0:
                break

            current = self.board[r][left]
            if current == 0:
                # Si hay piezas saltadas y no se ha encontrado una pieza vacía antes, se detiene la búsqueda
                if skipped and not last:
                    break
                # Si hay piezas saltadas, se agrega la posición de la última pieza y se continúa la búsqueda
                elif skipped:
                    moves[(r, left)] = last + skipped
                # Si no hay piezas saltadas, se agrega la posición de la última pieza y se detiene la búsqueda
                else:
                    moves[(r, left)] = last
                # Si se ha saltado alguna pieza, se busca si se pueden saltar más en ambas direcciones
                if last:
                    if step == -1:
                        row = max(r - 3, 0)
                    else:
                        row = min(r + 3, ROWS)
                    moves.update(
                        self._traverse_left(
                            r + step, row, step, color, left - 1, skipped=last
                        )
                    )
                    moves.update(
                        self._traverse_right(
                            r + step, row, step, color, left + 1, skipped=last
                        )
                    )
                break
            elif current.color == color:
                break
            else:
                last = [current]

            left -= 1

        return moves

    def _traverse_right(self, start, stop, step, color, right, skipped=[]):
        moves = {}
        last = []
        for r in range(start, stop, step):
            if right >= COLS:
                break

            current = self.board[r][right]
            if current == 0:
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, right)] = last + skipped
                else:
                    moves[(r, right)] = last

                if last:
                    if step == -1:
                        row = max(r - 3, 0)
                    else:
                        row = min(r + 3, ROWS)
                    moves.update(
                        self._traverse_left(
                            r + step, row, step, color, right - 1, skipped=last
                        )
                    )
                    moves.update(
                        self._traverse_right(
                            r + step, row, step, color, right + 1, skipped=last
                        )
                    )
                break
            elif current.color == color:
                break
            else:
                last = [current]

            right += 1

        return moves
