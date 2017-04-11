# -*- coding: utf-8 -*-
import pygame
import sys
import matplotlib.pyplot as plt

"""
与えられた座標列を表示する
キーボードの左右で履歴の再生，巻き戻し
"""

def draw(pos_list):
    (w, h) = (800, 800)
    (x, y) = (w//2, h//2)
    pygame.init()
    pygame.display.set_mode((w, h), 0, 32)
    screen = pygame.display.get_surface()
    ox, oy = x, y
    turn = 0

    while True:
        pygame.display.update()
        pygame.time.wait(30)
        screen.fill((0, 20, 0, 0))
        if turn < len(pos_list):
            x = ox + int(pos_list[turn][0])
            y = oy - int(pos_list[turn][1])
        pygame.draw.circle(screen, (0, 200, 0), (x, y), 5)
        pygame.draw.line(screen, (100, 200, 0), (x, y), (ox, oy))
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_LEFT] and turn > 0:
            turn -= 1
        if pressed[pygame.K_RIGHT] and turn < len(pos_list)-1:
            turn += 1

        for event in pygame.event.get():
            if is_exit(event):
                pygame.quit()
                return

def is_exit(event):
    return (
        event.type == pygame.KEYDOWN and
        event.key == pygame.K_ESCAPE or
        event.type == pygame.QUIT
    )

def draw_digit(data, size):
    plt.figure(figsize=(10, 10))
    X, Y = np.meshgrid(range(size), range(size))
    Z = data.reshape(size, size)
    Z = Z[::-1, :]
    plt.xlim(0, size-1)
    plt.ylim(0, size-1)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.show()


if __name__ == '__main__':
    from model import *
    (w, h) = (400, 400)
    (x, y) = (w//2, h//2)
    pygame.init()
    pygame.display.set_mode((w, h), 0, 32)
    screen = pygame.display.get_surface()
    env = Cource()
    env.reset()

    car = env.car
    ox, oy = x, y
    turn = 0

    while True:
        pygame.display.update()
        pygame.time.wait(500)
        screen.fill((0, 20, 0, 0))
        x =  int(car.x) + ox
        y = -int(car.y) + oy
        pygame.draw.circle(screen, (0, 200, 0), (x, y), 5)
        pygame.draw.line(screen, (100, 200, 0), (x, y), (ox, oy))
        command = 0
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_LEFT]:
            command |= car.OP_LT
        if pressed[pygame.K_RIGHT]:
            command |= car.OP_RT
        if pressed[pygame.K_z]:
            command |= car.OP_ACC
        if pressed[pygame.K_x]:
            command |= car.OP_BRK
        obs, rew, done, _ = env.step(command)
        # _sum = 0
        # for row in obs:
        #     for col in row:
        #         _sum += col
        # if abs(_sum) > 20:
        #     draw_digit(obs, obs.shape[0])
        #     car.dir += 0.3

        for event in pygame.event.get():
            if is_exit(event):
                pygame.quit()
                sys.exit()