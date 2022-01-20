"""
Traffic model kks, a multi-lane highway.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import animation
from functools import reduce


class Car:
    """A simple car. 创建新的车辆，单一。"""
    def __init__(self, pos=0, v=0, angle=0, lane=0):
        self.pos = pos  # position
        self.v = v  # velocity
        self.angle = angle  # This is used to a circular road
        self.r = lane + 1 # radius of the circular road


class Road:
    """A multi-lane road."""
    def __init__(self, max_v=2, road_length=50, max_cars=25, num_lane=2):
        # Road information
        self.road_length = road_length
        self.max_cars = max_cars
        self.num_lane = num_lane
        self.max_v = max_v

        # The information of car.
        # [[lane1_car1, lane1_car2, lane1_car3. ...]
        #  [lane2_car1, lane2_car2, lane2_car3,....]
        #  ....]
        self.tot_cars = 0  # the number of the cars on the highway
        self.num_car = 0  # the number of the cars on the lane
        self.carlist = np.ones((num_lane,max_car))*(-1)
        self.color = np.ones((num_lane,max_car))*(-1)
        self.anglelist = np.ones((num_lane,max_car))*(-1) # used to a circular raod
        self.r = np.ones((num_lane,max_car))*(-1) # used to a circular raod

    def calcu_num_car(self, lane):
        """Calculate the number of the cars on the lane."""
        self.num_car = 0
        for i in self.carlist[lane]:
            if i >= 0:
                self.num_car += 1

    def add_car(self, car, lane, mark):
        """Add car to the lane.
        There are two kind of the initial position.
        if mark = a:  The cars are evenly distributed on the road.
        if mark = b: The cars line up at the starting point of the road.
        car = (pos, vel, angle)."""
        self.calcu_num_car(lane)

        if self.num_car < self.max_cars:
            self.tot_cars += 1

            # The initial positions of the cars are evenly distributed on the road. 汽车起初位置，平均分布在道路上
            if mark == "a":
                car.pos = (self.road_length - self.num_car * self.road_length / self.max_cars) % self.road_length
                car.angle = self.num_car * 2 * np.pi / self.max_cars

            # The cars line up at the starting point of the road.
            elif mark == "b":
                car.pos = (self.road_length - self.num_car) % self.road_length
                car.angle = car.pos * 2 * np.pi / self.road_length

            self.color[lane,num_car+1] = num_car
            self.carlist[lane, num_car+1] = car
            self.anglelist[lane, num_car+1] = car.angle
            self.r[lane, num_car] = car.r

    def len_changing(self, car_index, car, lane, safe_d=1, pc=0.07, p=0.5, gc=3, delta1=1, delta2=1):
        # Calculate the distance between cars
        mark1 = False
        mark2 = False
        safe_l = False
        safe_r = False
        r_l = False
        l_r = False
        g_front_l = -1
        g_front_r = -1
        g_back_l = -1
        g_back_r = -1
        front_index_r = 0
        front_index_l = 0

        if lane + 1 < self.num_lane:  # There is a lane on the right
            mark1 = True
            self.calcu_num_car(lane + 1)
            while front_index_r < self.num_car:  # To find the front car on the right lane. 寻找右车道前车
                if self.carlist[lane + 1, front_index_r].pos <= car.pos:
                    front_index_r += 1
                else:
                    break
            g_front_r = (self.carlist[lane + 1, front_index_r].pos - car.pos - safe_d +
                         self.road_length) % self.road_length
            g_back_r = (car.pos - self.carlist[lane + 1, front_index_r + 1].pos - safe.d +
                        self.road_length) % self.road_length

        if lane > 0:  # There is a lane on the left.
            mark2 = True
            self.calcu_num_car(lane - 1)
            while front_index_l < self.num_car:  # To find the front car on the left lane. 寻找左车道前车
                if self.carlist[lane - 1, front_index_l].pos <= car.pos:
                    front_index_l += 1
                else:
                    break
            g_front_l = (self.carlist[lane - 1, front_index_l].pos - car.pos - safe_d +
                         self.road_length) % self.road_length
            g_back_l = (car.pos - self.carlist[lane - 1, front_index_l + 1].pos - safe.d +
                        self.road_length) % self.road_length

        # Rule 1, Safety Conditions
        if mark1 is True and g_front_r >= min(car.v, gc) and \
                g_back_r >= min(self.carlist[lane + 1, front_index_r + 1].v, gc):
            safe_r = True
        elif mark2 is True and g_front_l >= min(car.v, gc) and \
                g_back_l >= min(self.carlist[lane - 1, front_index_l + 1].v, gc):
            safe_l = True

        # Rule 2, Velocity Condition
        if mark1 is True and self.carlist[lane + 1, front_index_r].v >= self.carlist[lane, car_index - 1].v + delta1 or \
                self.carlist[lane + 1, front_index_r].v >= car.v + delta1:
            l_r = True
        elif mark2 is True and self.carlist[lane - 1, front_index_l].v >= self.carlist[lane, car_index - 1].v + delta1 or \
                car.v >= self.carlist[lane, car_index - 1].v:
            r_l = True

        # Random lan changing with probability pc. 随机变道
        if safe_r is True and l_r is True and random.random() <= pc:
            list_r = np.copy(self.carlist[lane + 1, front_index_r + 1:])
            list_o = np.copy(self.carlist[lane, car_index + 1:])
            self.carlist[lane + 1, front_index_r + 1] = car
            self.carlist[lane + 1, front_index_r + 2:] = list_r
            self.carlist[lane, car_index:] = list_o
        elif safe_l is True and r_l is True and random.random() <= pc:
            list_r = np.copy(self.carlist[lane - 1, front_index_l + 1:])
            list_o = np.copy(self.carlist[lane, car_index + 1:])
            self.carlist[lane - 1, front_index_l + 1] = car
            self.carlist[lane - 1, front_index_l + 2:] = list_r
            self.carlist[lane, car_index:] = list_o

    def update(self, lane=0, safe_d=1, pc=0.07, p=0.5, gc=3, delta1=1, delta2=1):
        """Update the position of the car.
        pc = the probability of overtaking/lane-changing.
        p = the probability of velocity slowdown.
        gc = lane change safety distance"""

        for i, car in enumerate(self.carlist[lane]):

            # Lane-changing
            self.len_changing(car_index=i, car=car, lane=lane, safe_d=safe_d, pc=pc, gc=gc, delta1=delta1, delta2=delta2)








