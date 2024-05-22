from pygame.locals import *
import pygame, asyncio
import sys
from constants import WIDTH, HEIGHT, WHITE, SQUARE, BLACK
from game import Game
from minmax.algorithm import minimax
import time


mainClock = pygame.time.Clock()
pygame.init()
pygame.display.set_caption("JUEGO DAMAS")
screen = pygame.display.set_mode((WIDTH, HEIGHT))


font = pygame.font.SysFont("arialblack", 15)


def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, 1, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)


click = False


def get_row_col_from_mouse(pos):
    x, y = pos
    row = y // SQUARE
    col = x // SQUARE
    return row, col


async def main():
    run = True
    game = Game(screen)

    while run:
        screen.fill((135, 206, 235))  # Cambiar el color a azul cielo
        draw_text("JUEGO DE DAMAS ", font, (0, 0, 0), screen, 175, 30)

        mx, my = pygame.mouse.get_pos()

        button_1 = pygame.Rect(150, 100, 200, 50)
        button_3 = pygame.Rect(150, 160, 200, 50)

        if button_1.collidepoint((mx, my)):
            if click:
                USUARIOvsIA(game)
        if button_3.collidepoint((mx, my)):
            if click:
                usuarioVSusuario(game)

        pygame.draw.rect(screen, (192, 192, 192), button_1)
        button_1_text = font.render("HUMANO vs IA", True, (0, 0, 0))
        button_1_text_rect = button_1_text.get_rect(center=button_1.center)
        screen.blit(button_1_text, button_1_text_rect)

        pygame.draw.rect(screen, (192, 192, 192), button_3)
        button_3_text = font.render("HUMANO vs HUMANO", True, (0, 0, 0))
        button_3_text_rect = button_3_text.get_rect(center=button_3.center)
        screen.blit(button_3_text, button_3_text_rect)

        click = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = get_row_col_from_mouse(pos)
                game.select(row, col)

                if event.button == 1:
                    click = True
        await asyncio.sleep(0)
        pygame.display.update()

    pygame.quit()



def USUARIOvsIA(game):
    running = True
    population_size = 4
    num_generations = 1
    user_move_start_time = None
    user_move_time_limit = 10  # el usuario tiene 10 segundos para hacer un movimiento

    while running:
        if game.turn == WHITE:
            value, new_board = minimax(
                game.getBoard(), 2, WHITE, game, population_size, num_generations
            )
            game.ai_move(new_board)
            pygame.display.update()
            user_move_start_time = (
                time.time()
            )  
        # interfaz para ganador
        if game.winner() is not None:
            pygame.display.update()

            winner = game.winner()
            print(winner)
            font = pygame.font.SysFont(None, 48)
            text = font.render("El ganador es: " + str(winner), True, (255, 255, 255))
            text_rect = text.get_rect(center=screen.get_rect().center)
            screen.blit(text, text_rect)
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = get_row_col_from_mouse(pos)
                game.select(row, col)

                # si el usuario hace un movimiento, reiniciar el temporizador
                user_move_start_time = time.time()

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

        # si el usuario se tarda demasiado, terminar el juego
        if (
            user_move_start_time is not None
            and time.time() - user_move_start_time > user_move_time_limit
        ):
            
            
            font = pygame.font.SysFont(None, 48)
            text = font.render("Perdio Turno: " , True, (255, 255, 255))
            text_rect = text.get_rect(center=screen.get_rect().center)
            screen.blit(text, text_rect)
            pygame.time.delay(500)
            game.turn=WHITE
            

        game.update()
        pygame.display.update()
        mainClock.tick(60)

    pygame.quit()


def usuarioVSusuario(game):
    running = True

    while running:
        if game.turn == WHITE:
            pygame.display.update()

        # interfaz para ganador
        if game.winner() is not None:
            pygame.display.update()

            winner = game.winner(WHITE)
            print(winner)
            font = pygame.font.SysFont(None, 48)
            text = font.render("El ganador es: " + winner, True, (255, 255, 255))
            text_rect = text.get_rect(center=screen.get_rect().center)
            screen.blit(text, text_rect)
            running = False
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = get_row_col_from_mouse(pos)
                game.select(row, col)

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
        game.update()
        pygame.display.update()
        mainClock.tick(60)

    pygame.quit()


def iaVSia():
    running = True
    while running:
        screen.fill((0, 0, 0))

        draw_text("Proximamente chavalin", font, (255, 255, 255), screen, 20, 20)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

        pygame.display.update()
        mainClock.tick(60)


asyncio.run(main())
