# -*- encoding: utf-8 -*-
import math
from gym import spaces
import numpy as np
from scipy import ndimage

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

CAR_SITE_R = 20
FIELD_SIZE = 200
MAP_SIZE = (FIELD_SIZE + 50)*2
OFFSET = MAP_SIZE//2

class Cource:
    """
    Carが動くフィールドを表す(2次元空間)
    """
    ACTIONS = 11     # 取れる行動の種類数
    OBS_SIZE = (CAR_SITE_R*2)**2
    FIELD_R = FIELD_SIZE
    def __init__(self):
        self.turn = 0
        self.action_space = spaces.Discrete(self.ACTIONS)

    def reset(self):
        """
        環境の初期化をする
        """
        self.field = np.zeros((MAP_SIZE, MAP_SIZE))
        self._rewrite_map()

        car_dir = np.random.rand() * math.pi*2 - math.pi
        pos_dir = np.random.rand() * math.pi*2 - math.pi
        dist = np.random.rand() * 3 + self.FIELD_R/2
        x = np.cos(pos_dir)*dist
        y = np.sin(pos_dir)*dist
        self.car = Car(x, y, car_dir)
        # self.car = Car(8, 0, math.pi/2)
        self.turn = 0
        return self.car.observe(self.field)

    def render(self):
        """
        ステップごとの描画関数
        """
        print('turn: %3d; car=%s' % (self.turn, str(self.car)))

    def step(self, action):
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
        pre_vec = self.car.get_vec()
        self.car.update(action)
        dist = self.car._dist()
        reward = 0
        if dist >= self.FIELD_R:
            self.car.force_move(dist, dist-self.FIELD_R)
        done = self.turn >= 2000
        reward += math.sqrt(dist2(self.car.get_vec(), pre_vec))/self.FIELD_R
        info = str(self.car)
        self.turn += 1
        return self.car.observe(self.field), reward, done, info

    def _rewrite_map(self):
        shape = self.field.shape
        sq_size = self.FIELD_R**2
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (i-OFFSET)**2 + (j-OFFSET)**2 > sq_size:
                    self.field[i][j] = 1

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


class Car:
    """
    車を表す
    2次元平面上を走ることができ，アクセル，方向転換ができる
    """
    OP_ACC = 1    # アクセル操作
    OP_BRK = 2
    OP_RT = 4     # 右ハンドル操作
    OP_LT = 8     # 左ハンドル操作
    HND_GRD = 0.1 # 方向転換の度合い
    SPEED = 0.15   # アクセルの加速度
    SPEED_DEC = 0.9 # 減速度
    MAX_SPEED = 3

    CAR_R = 5
    SITE_R = CAR_SITE_R
    SITE_R_LARGE = int(SITE_R*1.5)
    SITE_R_DIFF = SITE_R_LARGE - SITE_R
    def __init__(self, _x, _y, _dir):
        """
        現在位置と向き(絶対角度)を持つ
        """
        self.x = _x
        self.y = _y
        self.v = 0
        self.dir = _dir

    def observe(self, _map):
        """
        周囲SITE_Rの景色を返す
        """
        x = int(self.x)
        y = int(self.y)
        sub_map = _map[
            OFFSET+x-self.SITE_R_LARGE : OFFSET+x+self.SITE_R_LARGE,
            OFFSET+y-self.SITE_R_LARGE : OFFSET+y+self.SITE_R_LARGE
        ]
        sub_map = ndimage.interpolation.rotate(
            sub_map,
            math.degrees(-self.dir+math.pi/2),
            reshape=False
        )[
            self.SITE_R_DIFF : self.SITE_R_DIFF + self.SITE_R*2,
            self.SITE_R_DIFF : self.SITE_R_DIFF + self.SITE_R*2
        ]
        return np.array(sub_map, dtype=np.float32)

    def _dist(self):
        return math.sqrt(self.x**2 + self.y**2)

    def _angle(self):
        return angle((-self.x, -self.y), (math.cos(self.dir), math.sin(self.dir)))

    def get_vec(self):
        return (self.x, self.y)

    def collide(self, car):
        return (self.x-car.x)**2 + (self.y-car.y)**2 < (self.CAR_R*2) ** 2

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
            if self.dir >= math.pi:
                self.dir -= math.pi*2
        elif op & self.OP_LT:
            self.dir -= self.HND_GRD
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
