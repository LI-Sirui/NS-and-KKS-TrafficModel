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
    def __init__(self, max_v=2, road_length=50, max_car=25, num_lane=2):
        # Road information
        self.road_length = road_length
        self.max_cars = max_car
        self.num_lane = num_lane
        self.max_v = max_v

        # The information of car.
        # [ [lane1_car1, lane1_car2, lane1_car3. ...],
        #   [lane2_car1, lane2_car2, lane2_car3,....],
        #  .... ]
        self.tot_cars = 0  # the number of the cars on the highway
        self.carlist = [[] for i in range(num_lane)]
        self.color = [[] for i in range(num_lane)]
        self.anglelist = [[] for i in range(num_lane)] # used to a circular road
        self.r = [[] for i in range(num_lane)]  # used to a circular road

        # pick
        self.num_car = 0  # the number of the cars on the lane, pick, change always
        # self.lane = 0  # pick of the lane change always

    def add_car(self, car, lane, mark):
        """Add car to the lane.
        There are two kind of the initial position.
        if mark = a:  The cars are evenly distributed on the road.
        if mark = b: The cars line up at the starting point of the road.
        car = (pos, vel, angle)."""

        if len(self.carlist[lane]) < self.max_cars:
            self.tot_cars += 1

            # The initial positions of the cars are evenly distributed on the road. 汽车起初位置，平均分布在道路上
            if mark == "a":
                car.pos = (self.road_length - self.num_car * self.road_length / self.max_cars) % self.road_length
                car.angle = self.num_car * 2 * np.pi / self.max_cars

            # The cars line up at the starting point of the road.
            elif mark == "b":
                car.pos = (self.road_length - self.num_car) % self.road_length
                car.angle = car.pos * 2 * np.pi / self.road_length

            self.color[lane].append(len(self.carlist[lane]))
            self.carlist[lane].append(car)
            self.anglelist[lane].append(car.angle)
            self.r[lane].append(car.r)

    def infor_len_changing(self, carlist, car_index, car, lane, safe_d=1, pc=0.07, gc=3, delta1=1, delta2=1):
        """Determine lane change conditions and implement lane changes."""
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
        back_index_r = 0
        back_index_l = 0
        back_index = 0
        front_index_r = 0

        if lane + 1 < self.num_lane:  # There is a lane on the right
            mark1 = True

            while back_index_r < len(carlist[lane+1])-1:  # To find the front car on the right lane. 寻找右车道前车
                if carlist[lane + 1][back_index_r].pos > car.pos:
                    back_index_r += 1
                else:
                    break
            if back_index != 0:
                front_index_r = back_index_r - 1
                if back_index_r == len(carlist[lane+1])-1:
                    if carlist[lane+1][front_index_r].pos > car.pos:
                        front_index_r = back_index_r
                        back_index_r = 0
                        Problem here!!!!!!!!!!!!!




            g_front_r = (carlist[lane + 1][front_index_r].pos - car.pos - safe_d +
                         self.road_length) % self.road_length
            g_back_r = (car.pos - carlist[lane + 1][front_index_r + 1].pos - safe_d +
                        self.road_length) % self.road_length

        if lane > 0:  # There is a lane on the left.
            mark2 = True
            while front_index_l < len(carlist[lane-1]):  # To find the front car on the left lane. 寻找左车道前车
                if carlist[lane - 1][front_index_l].pos <= car.pos:
                    front_index_l += 1
                else:
                    break
            g_front_l = (carlist[lane - 1][front_index_l].pos - car.pos - safe_d +
                        self.road_length) % self.road_length
            g_back_l = (car.pos - carlist[lane - 1][front_index_l + 1].pos - safe_d +
                        self.road_length) % self.road_length

        while front_index < len(carlist[lane]):  # To find the front car on the orig lane
            if carlist[lane][front_index].pos <= car.pos:
                front_index += 1
            else:
                break

        # Rule 1, Safety Conditions
        if mark1 is True and g_front_r >= min(car.v, gc) and \
                g_back_r >= min(carlist[lane + 1][front_index_r + 1].v, gc):
            safe_r = True
        elif mark2 is True and g_front_l >= min(car.v, gc) and \
                g_back_l >= min(carlist[lane - 1][front_index_l + 1].v, gc):
            safe_l = True

        # Rule 2, Velocity Condition
        if mark1 is True and carlist[lane + 1][front_index_r].v >= carlist[lane][car_index - 1].v + delta1 \
                or carlist[lane + 1][front_index_r].v >= car.v + delta1:
            l_r = True
        elif mark2 is True and carlist[lane - 1][front_index_l].v >= carlist[lane][car_index - 1].v + delta2 \
                or car.v >= carlist[lane][car_index - 1].v:
            r_l = True

        # Random lan changing with probability pc. 随机变道
        if safe_r is True and l_r is True and random.random() < pc:
            return "l_to_r", front_index_r

        elif safe_l is True and r_l is True and random.random() <= pc:
            return "r_to_l", front_index_l

        else:
            return "No", front_index

    def lan_changing(self, safe_d=1, pc=0.07, gc=3, delta1=1, delta2=1):
        """Run lane changing for the highway."""
        new_carlist = self.carlist[:]
        new_anglelist = self.anglelist[:]
        ch_ls = [[] for j in range(self.num_lane)]
        f_index_ls = [[] for j in range(self.num_lane)]

        for lane, lanels in enumerate(self.carlist):
            for i, car in enumerate(self.carlist[lane]):
                changing, front_index = self.infor_len_changing(carlist=new_carlist, car_index=i, car=car, lane=lane,
                                                               safe_d=safe_d, pc=pc, gc=gc, delta1=delta1,
                                                               delta2=delta2)
                ch_ls[lane].append(changing)
                f_index_ls[lane].append(front_index)

        new_f_index_ls = f_index_ls[:]
        for lane in range(len(ch_ls)):  # each lane
            for i, m in enumerate(ch_ls[lane]):  # each car
                o_lane = lane
                front_index = f_index_ls[lane][i]
                if m == "l_to_r":  # change to the right lane, 向右变道
                    lane = lane + 1
                elif m == "r_to_l":  # change to the left lane, 向左变道
                    lane = lane - 1
                elif m == "No":  # Not change, 不变道
                    pass
                else:
                    raise NameError('Error for lan-changing!')

                # update the car list on the highway
                if m != "No":  # if lane changing
                    new_carlist[lane][front_index+1] = self.carlist[o_lane][i]
                    new_anglelist[lane][front_index+1] = self.anglelist[o_lane][i]

                    new_carlist[lane][front_index+2 :] = self.carlist[lane][front_index+1 :]
                    new_anglelist[lane][front_index+2 :] = self.anglelist[lane][front_index+1 :]

                    new_carlist[o_lane][i :] = self.carlist[o_lane][i+1 :]
                    new_anglelist[o_lane][i :] = self.anglelist[o_lane][i+1 :]

        self.carlist = new_carlist
        self.anglelist = new_anglelist

    def update(self, safe_d=1, pc=0.07, p=0.5, gc=3, delta1=1, delta2=1, k=3):
        """Update the position of the car.
        pc = the probability of overtaking/lane-changing.
        p = the probability of velocity slowdown.
        gc = lane change safety distance
        G = k*v = the safe distance, set G=v """
        # self.lane = lane
        self.lan_changing(safe_d=safe_d, pc=pc, gc=gc, delta1=delta1,
                                                               delta2=delta2)

        for lane in range(len(self.carlist)):
            for i, car in enumerate(self.carlist[lane]):
                # distance from the car in front
                g = (self.carlist[lane][i-1].pos - car.pos - safe_d + self.road_length) % self.road_length

                # Accelerate
                G = k * car.v
                if g <= G:
                    car.v = car.v + np.sign(self.carlist[lane][i-1].v - car.v)
                else:
                    car.v = min(car.v+1, self.max_v)

                # Deceleration in relation to safety distance
                if car.v >= 0 and g > 0:
                    car.v = min(car.v, g)
                else:
                    raise NameError('Error for g or car.v!')

                # Random slow down
                if random.random() < p:
                    car.v = max(car.v-1, 0)

        # update position of the cars
        self.update_pos()


    def update_pos(self):
        """Update position of the cars on the lane"""
        for lane_index, lane in enumerate(self.carlist):
            for i, car in enumerate(lane):
                car.pos = (car.pos + car.v) % self.road_length
                self.anglelist[lane_index][i] = car.pos * 2 * np.pi / self.road_length

    def density(self):
        """Calculate the car density."""
        pass

    def flow_rate(self):
        """Calculate the flow rate of the highway"""
        pass


def animate(frameNr, ax, road, p, pc):
    ax.clear()
    ax.set_rlim(0, 1.5)
    ax.axis('off')
    road.update(p=p, pc=pc)
    return ax.scatter(road.anglelist, road.r, c=road.color),


class Simulation:
    """Simulation traffic."""
    def __init__(self, road, pc, p, mark="b"):
        self.road =road
        self.p = p
        self.pc = pc
        self.mark = mark

    def run_animate(self, tot_car=25, mark="a", time=100, stepsperframe=1, title="Simulation", save=False):
        """Run the traffic model animation."""
        for i in range(tot_car):
            lane = random.randint(0, self.road.num_lane-1)
            self.road.add_car(Car(), lane=lane, mark=mark)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.axis('off')
        ax.set_title(title, va='bottom')
        numframes = int(time / stepsperframe)
        anim = animation.FuncAnimation(fig, animate, fargs=[ax, self.road, self.p, self.pc],
                                       frames=numframes, interval=50, blit=True, repeat=False, save_count=numframes)
        if save:
            writergif = animation.PillowWriter(fps=20)
            anim.save("car.gif", writer=writergif)
        else:
            plt.show()

def main():
    p = 0.5
    pc = 0.5
    road_length = 50
    max_v = 2
    num_lane = 2
    road = Road(max_v=max_v, road_length=road_length,max_car=25, num_lane=num_lane)
    simulation = Simulation(road=road, p=p, pc=pc, mark="b")
    simulation.run_animate(tot_car=25, mark="b", time=100, stepsperframe=1, title="Simulation")


if __name__ == '__main__':
    main()






