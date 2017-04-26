# -*- coding: utf-8 -*-
import pygame
import sys
import matplotlib.pyplot as plt

"""
与えられた座標列を表示する
キーボードの左右で履歴の再生，巻き戻し
"""

def print_info(str_list, screen):
    # フォントの作成
    sysfont = pygame.font.SysFont(None, 20)
    for i, row in enumerate(str_list):
        # テキストを描画したSurfaceを作成
        string = sysfont.render(row, False, (200,200,200))
        screen.blit(string, (10, i*20))


def sub_vec(a, b, val):
    c = (-a[0]+b[0], -a[1]+b[1])
    return (int(a[0]+c[0]*val), int(a[1]+c[1]*val))

CAR_R = 8

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
        pygame.draw.circle(screen, (0, 200, 0), (w//2, h//2), 60, 1)
        if turn < len(pos_list):
            for pos in pos_list[turn]:
                x = ox + int(pos[0])
                y = oy - int(pos[1])
                pygame.draw.circle(screen, (0, 200, 0), (x, y), CAR_R)
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

    cars = env.cars
    car = cars[0]
    ox, oy = x, y
    turn = 0

    actions = [0]*len(cars)

    while True:
        pygame.display.update()
        pygame.time.wait(33)
        screen.fill((0, 20, 0, 0))

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
        actions[0] = command
        obs_list = env.step(actions)

        message = []

        pygame.draw.circle(screen, (0, 200, 0), (w//2, h//2), env.FIELD_R, 1)
        for c in cars:
            x =  int(c.x) + ox
            y = -int(c.y) + oy
            pygame.draw.circle(screen, (0, 200, 0), (x, y), CAR_R)
            # pygame.draw.line(screen, (100, 200, 0), (x, y), (ox, oy))
            message.append(str(obs_list[c.id][0][0]))
            for i, radar in enumerate(c.get_radar_lines()):
                p1 = (int(radar[0][0]+ox), int(-radar[0][1]+oy))
                p2 = (int(radar[1][0]+ox), int(-radar[1][1]+oy))
                pygame.draw.line(screen, (200, 200, 0), p1, p2)
                val_f = obs_list[c.id][0][0][i]
                pygame.draw.circle(screen, (200, 0, 0), sub_vec(p1, p2, 1-val_f), 3)
                val_c = obs_list[c.id][0][1][i]
                pygame.draw.circle(screen, (0, 0, 200), sub_vec(p1, p2, 1-val_c), 3)

        print_info(message, screen)
        # obs, rew, done, _ = env.step(actions)
        # _sum = 0
        # for row in obs:
        #     for col in row:
        #         _sum += col
        # if abs(_sum) > 20:
        obs = obs_list[0][0]
        # draw_digit(obs, obs.shape[0])
        #     car.dir += 0.3

        for event in pygame.event.get():
            if is_exit(event):
                pygame.quit()
                sys.exit()
