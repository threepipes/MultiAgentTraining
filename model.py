# -*- encoding: utf-8 -*-
import math
from gym import spaces
import numpy as np
import shapely.geometry as sg
import shapely.affinity as sa

"""
2次元フィールド上で車を動かすシミュレーションのモデル
"""

def angle(a, b):
    dist_a = dist(a)
    dist_b = dist(b)
    if abs(a[0]-b[0]) < 1e-2 and abs(a[1]-b[1]) < 1e-2:
        return 0
    try:
        return math.acos((a[0]*b[0] + a[1]*b[1]) / (dist_a*dist_b)) \
           * np.sign(cross(a, b))
    except:
        print(a)
        print(b)
        raise


def dist(a):
    return math.sqrt(a[0]**2 + a[1]**2)


def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]


def same(a, b):
    return abs(a[0]-b[0]) < 1e-8 and abs(a[1]-b[1]) < 1e-8

CAR_SITE_R = 30
FIELD_SIZE = 60
MAP_SIZE = (FIELD_SIZE + 50)*2
OFFSET = MAP_SIZE//2

class Cource:
    """
    Carが動くフィールドを表す(2次元空間)
    """
    ACTIONS = 11     # 取れる行動の種類数
    OBS_SIZE = (CAR_SITE_R*2)**2
    FIELD_R = FIELD_SIZE
    N_AGENTS = 4
    def __init__(self):
        self.turn = 0
        self.action_space = spaces.Discrete(self.ACTIONS)
        self.field = np.zeros((MAP_SIZE, MAP_SIZE))
        self.circle = sg.Point(0, 0).buffer(self.FIELD_R).boundary

    def reset(self):
        """
        環境の初期化をする
        """
        self.cars = []

        for i in range(self.N_AGENTS):
            car_dir = np.random.rand() * math.pi*2 - math.pi
            pos_dir = np.random.rand() * math.pi*2 - math.pi
            # pos_dir = np.random.rand() / 5
            dist = i * 15 + self.FIELD_R/2
            x = np.cos(pos_dir)*dist
            y = np.sin(pos_dir)*dist
            self.cars.append(Car(x, y, car_dir, i))
        self.turn = 0
        return self.get_observes()

    def get_observes(self):
        obs = []
        for car in self.cars:
            obs.append(car.observe(self.field))
        return obs

    def render(self):
        """
        ステップごとの描画関数
        """
        info = 'turn: %3d\n' % self.turn
        for i, car in enumerate(self.cars):
            info += 'car %d: %s\n' % (i, str(car))
        print(info)

    def step(self, actions):
        """
        agentが選んだ行動が与えられるので
        環境を変更し，観察値や報酬を返す

        今回は，actionに応じて車を動かし，
        フィールド内で動いた量を評価する

        :param int action: どの行動を選んだか
        :return:
            observe: numpy array: 環境の観察値を返す
            reward : float      : 報酬
            done   : boolean    : 終了したか否か
            info   : str (自由?): (デバッグ用などの)情報
        """
        if len(actions) != len(self.cars):
            print('Action number is wrong.')
            return
        result = []
        for action, car in zip(actions, self.cars):
            pre_vec = car.get_vec()
            car.update(action)
            dist = car._dist()
            reward = 0
            if dist + car.CAR_R >= self.FIELD_R:
                car.force_move(dist, dist + car.CAR_R - self.FIELD_R)
                reward -= 1
            done = self.turn >= 2000
            reward += math.sqrt(dist2(car.get_vec(), pre_vec)) * dist / self.FIELD_R
            info = str(car)
            for other in self.cars:
                if other.id == car.id:
                    continue
                if car.collide(other):
                    reward -= (car.CAR_R * 2 - car.dist_car(other)) / car.CAR_R
            result.append([None, reward, done, info])

        for i, car in enumerate(self.cars):
            result[i][0] = car.observe(self, self.cars)
            result[i] = (result[i][0], result[i][1], result[i][2], result[i][3])

        self.turn += 1
        return result

    def _calc_angle_diff(self, a, b):
        if same(a, b):
            return 0
        return math.acos((a[0]*b[0] + a[1]*b[1]) / (self._dist(a)*self._dist(b)))

    def _dist(self, a):
        return math.sqrt(a[0]**2 + a[1]**2)

    def get_action_space(self):
        """
        :return: Descrete: とれる行動の種類数を返す
        """
        return self.action_space.sample

    def get_vecs(self):
        vecs = []
        for car in self.cars:
            vecs.append(car.get_vec())
        return vecs


class Radar:
    RADAR_N = 5
    RADAR_LEN = 100
    VISION = 100
    def __init__(self, car):
        self.car = car
        _dir = - self.VISION / 2
        self.lines = []
        car_dir = car.dir
        line_base = sg.LineString([(0, 0), (100, 0)])
        for i in range(self.RADAR_N):
            line = sa.rotate(line_base, _dir + car_dir, origin=(0, 0))
            _dir += self.VISION / (self.RADAR_N - 1)
            self.lines.append(line)

    def rotate(self, _dir):
        car_pos = (self.car.x, self.car.y)
        for i in range(self.RADAR_N):
            self.lines[i] = sa.rotate(self.lines[i], _dir, origin=car_pos)

    def reset(self):
        self.dist_field = [0] * self.RADAR_N
        self.dist_car = [0] * self.RADAR_N

    def intersect(self, dist, other):
        circle = other.circle
        center = self.car.center
        for i, line in enumerate(self.lines):
            for pos in circle.intersection(line):
                d = 1 - center.distance(pos) / self.RADAR_LEN
                dist[i] = max(dist[i], d)

    def find_intersects(self, field, car_list):
        self.intersect(self.dist_field, field)
        for _car in car_list:
            if _car.id == self.car.id:
                continue
            self.intersect(self.dist_car, _car)
        return [self.dist_field, self.dist_car]


class Car:
    """
    車を表す
    2次元平面上を走ることができ，アクセル，方向転換ができる
    """
    OP_ACC = 1    # アクセル操作
    OP_BRK = 2
    OP_RT = 4     # 右ハンドル操作
    OP_LT = 8     # 左ハンドル操作
    HND_GRD = 0.3 # 方向転換の度合い
    SPEED = 0.15   # アクセルの加速度
    SPEED_DEC = 0.7 # 減速度
    MAX_SPEED = 2

    CAR_R = 8
    SITE_R = CAR_SITE_R
    SITE_R_LARGE = int(SITE_R*1.5)
    SITE_R_DIFF = SITE_R_LARGE - SITE_R
    def __init__(self, _x, _y, _dir, _id):
        """
        現在位置と向き(絶対角度)を持つ
        """
        self.id = _id
        self.x = _x
        self.y = _y
        self.v = 0
        self.dir = _dir

        self.radar = Radar(self)
        self.center = sg.Point(_x, _y)
        self.circle = self.center.buffer(self.CAR_R).boundary

    def observe(self, field, car_list):
        """
        レーダーの観察を返す
        """
        self.radar.reset()
        dists = self.radar.find_intersects(field, car_list)
        return np.array(dists, dtype=np.float32)

    def _dist(self):
        return math.sqrt(self.x**2 + self.y**2)

    def _angle(self):
        return angle((-self.x, -self.y), (math.cos(self.dir), math.sin(self.dir)))

    def get_vec(self):
        return (self.x, self.y)

    def collide(self, car):
        return (self.x-car.x)**2 + (self.y-car.y)**2 < (self.CAR_R*2) ** 2

    def dist_car(self, car):
        return math.sqrt((self.x-car.x)**2 + (self.y-car.y)**2)

    def update(self, action):
        """
        actionに応じて車を移動させる
        下位1,2ビットがアクセル,ブレーキ
        3,4ビット目でハンドル操作
        """
        self._op_handle(action)
        if action & self.OP_ACC:
            self._op_accel(1)
        elif action & self.OP_BRK:
            self._op_accel(-1)
        else:
            self._speed_down()

        self.x += math.cos(self.dir) * self.v
        self.y += math.sin(self.dir) * self.v

    def _speed_down(self):
        self.v *= self.SPEED_DEC
        if abs(self.v) < 1e-4:
            self.v = 0

    def _op_accel(self, brake):
        """
        :brake int: ブレーキなら-1，進むなら1
        """
        self.v += self.SPEED * brake
        # 速度制限
        if self.v > self.MAX_SPEED:
            self.v = self.MAX_SPEED
        elif self.v < -self.MAX_SPEED:
            self.v = -self.MAX_SPEED

    def _op_handle(self, op):
        if op & self.OP_RT:
            self.dir += self.HND_GRD
            self.radar.rotate(self.HND_GRD)
            if self.dir >= math.pi:
                self.dir -= math.pi*2
        elif op & self.OP_LT:
            self.dir -= self.HND_GRD
            self.radar.rotate(-self.HND_GRD)
            if self.dir < -math.pi:
                self.dir += math.pi*2

    def force_move(self, dist, force):
        """
        (0, 0)方向にforce動かす
        """
        self.x -= self.x/dist * force
        self.y -= self.y/dist * force

    def __str__(self):
        return '(%f, %f: %f)' % (self.x, self.y, self.dir)
