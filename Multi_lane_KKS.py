"""
Traffic model kks, a multi-lane highway.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import animation
from functools import reduce
import copy

# random.seed(1)
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
        # self.num_car = 0  # the number of the cars on the lane, pick, change always
        # self.lane = 0  # pick of the lane change always

    def add_car(self, car, lane, mark="b"):
        """Add car to the lane.
        There are two kind of the initial position.
        if mark = a:  The cars are evenly distributed on the road.
        if mark = b: The cars line up at the starting point of the road.
        car = (pos, vel, angle)."""

        if len(self.carlist[lane]) < self.max_cars:
            self.tot_cars += 1

            # The initial positions of the cars are evenly distributed on the road. 汽车起初位置，平均分布在道路上
            if mark == "a":
                car.pos = (self.road_length - len(self.carlist[lane]) * self.road_length / self.max_cars) % self.road_length
                car.angle = len(self.carlist[lane]) * 2 * np.pi / self.max_cars

            # The cars line up at the starting point of the road.
            elif mark == "b":
                car.pos = (self.road_length - len(self.carlist[lane])) % self.road_length
                car.angle = car.pos * 2 * np.pi / self.road_length

            color_number = len(self.carlist[lane])
            self.color[lane].append(color_number)
            self.carlist[lane].append(car)
            self.anglelist[lane].append(car.angle)
            self.r[lane].append(car.r)

            # for c in self.carlist[0]:
                # print(c.pos)

    def find_car_index_leftright(self, carlist, car, lane):
        """To find the index of the front car and back car on the right, left lane."""
        mark1=False
        mark2=False
        back_index_r = 0
        back_index_l = 0
        front_index_l = 0
        front_index_r = 0

        # Find the index of car on the right lane.
        if lane + 1 < self.num_lane:  # There is a lane on the right
            mark1 = True
            if len(carlist[lane + 1]) > 1:  # There are least 2 car on the right lane.
                while back_index_r < len(carlist[lane + 1]) - 1:  # To find the front car on the right lane. 寻找右车道前车
                    if carlist[lane + 1][back_index_r].pos > car.pos:
                        back_index_r += 1
                    else:
                        break

                if back_index_r != 0:
                    front_index_r = back_index_r - 1
                    if back_index_r == len(carlist[lane + 1]) - 1:  # 如果back是第一辆车
                        if carlist[lane + 1][back_index_r - 1].pos > carlist[lane + 1][back_index_r].pos > car.pos:
                            # 最后一辆车在车前，最后一辆车为前车，第一辆车为后车。后车（0号车）的位置大于car，即最大为roadLength-1，
                            front_index_r = back_index_r
                            back_index_r = 0
                        elif carlist[lane + 1][back_index_r].pos > carlist[lane + 1][back_index_r - 1].pos > car.pos:
                            # 最后一辆车在车后，为后车。倒数第二辆车为前车。后车位置大于0号车且小于RL，但前车位置小于0号车。
                            front_index_r = back_index_r - 1
                        elif carlist[lane + 1][back_index_r].pos <= car.pos:
                            # 最后一辆车在车后，为后车。倒数第二辆车为前车。0号车的位置在car前方，即大于car的位置，0号车位置最大为RL-1
                            pass
                        # else:
                            # for c in carlist[lane+1]:
                                # print(c.pos)
                            # print(back_index_r, back_index_r-1)
                            # print(carlist[lane+1][back_index_r].pos, car.pos, carlist[lane+1][back_index_r-1].pos)
                            # raise NameError('Error for changing-infor av the index of the car on the right lane.')
                elif back_index_r == 0:
                    front_index_r = len(carlist[lane + 1]) - 1
                else:
                    raise NameError('Error for back index of right.')

            elif len(carlist[lane + 1]) == 0:
                back_index_r, front_index_r = None, None

        # To find the index of car on the left lane.
        if lane > 0:  # There is a lane on the left.
            mark2 = True
            if len(carlist[lane - 1]) > 1:  # There is least two cars on the left lane.
                while back_index_l < len(carlist[lane - 1]) - 1:  # To find the front car on the left lane. 寻找左车道前车
                    if carlist[lane - 1][back_index_l].pos > car.pos:
                        back_index_l += 1
                    else:
                        break
                if back_index_l != 0:
                    front_index_l = back_index_l - 1
                    if back_index_l == len(carlist[lane - 1]) - 1:
                        if carlist[lane - 1][back_index_l - 1].pos > carlist[lane - 1][back_index_l].pos > car.pos:
                            # 最后一辆车在车前，最后一辆车为前车，第一辆车为后车。后车（0号车）的位置大于car，即最大为roadLength-1，
                            front_index_l = back_index_l
                            back_index_l = 0
                        elif carlist[lane - 1][back_index_l].pos > carlist[lane - 1][back_index_l - 1].pos > car.pos:
                            # 最后一辆车在车后，为后车。倒数第二辆车为前车。后车位置大于0号车且小于RL，但前车位置小于0号车。
                            front_index_l = back_index_l - 1
                        elif carlist[lane - 1][back_index_l].pos <= car.pos:
                            pass
                        # else:
                            # posls = [[], []]
                            # for l in range(2):
                            #     for c in self.carlist[l]:
                            #         posls[l].append(c.pos)
                            # print(posls)
                            # print(lane-1, back_index_l, carlist[lane-1][back_index_l].pos, carlist[lane-1][back_index_l-1].pos, car.pos)
                            # raise NameError('Error for changing infor av the back car on the left.')

                elif back_index_l == 0:
                    front_index_l = len(carlist[lane - 1]) - 1
                else:
                    raise NameError('Error for back index of left.')

            elif len(carlist[lane - 1]) == 0:
                back_index_l, front_index_l = None, None

        return mark1, mark2, front_index_r, back_index_r, front_index_l, back_index_l

    def infor_len_changing(self, carlist, car_index, car, lane, safe_d=1, pc=0.07, gc=3, delta1=1, delta2=1):
        """Determine lane change conditions and implement lane changes."""
        safe_l = False
        safe_r = False
        r_l = False
        l_r = False
        g_front_l = -1
        g_front_r = -1
        g_back_l = -1
        g_back_r = -1

        mark1, mark2, front_index_r, back_index_r, front_index_l, back_index_l = self.find_car_index_leftright(carlist,
                                                                                                               car, lane)
        # Calculate the distance between cars
        if mark1 is True and front_index_r is not None and back_index_r is not None:
            g_front_r = (carlist[lane + 1][front_index_r].pos - car.pos - safe_d + self.road_length) % self.road_length
            g_back_r = (car.pos - carlist[lane + 1][back_index_r].pos - safe_d + self.road_length) % self.road_length

        if mark2 is True and front_index_l is not None and back_index_l is not None:
            g_front_l = (carlist[lane - 1][front_index_l].pos - car.pos - safe_d + self.road_length) % self.road_length
            g_back_l = (car.pos - carlist[lane - 1][back_index_l].pos - safe_d + self.road_length) % self.road_length

        # Rule 1, Safety Conditions
        if mark1 is True:
            if g_front_r >= min(car.v, gc) and g_back_r >= min(carlist[lane + 1][back_index_r].v, gc):
                safe_r = True
            elif front_index_r is None:
                safe_r = True

        elif mark2 is True:
            if g_front_l >= min(car.v, gc) and g_back_l >= min(carlist[lane - 1][back_index_l].v, gc):
                safe_l = True
            elif front_index_l is None:
                safe_l = True

        # Rule 2, Velocity Condition
        if mark1 is True:
            if front_index_r is None:
                l_r = True
            elif carlist[lane + 1][front_index_r].v >= carlist[lane][car_index - 1].v + delta1 or \
                    carlist[lane + 1][front_index_r].v >= car.v + delta1:
                l_r = True

        elif mark2 is True:
            if car.v >= carlist[lane][car_index - 1].v:
                if front_index_l is None:  # No car on the left lane.
                    r_l = True
                elif carlist[lane - 1][front_index_l].v >= carlist[lane][car_index - 1].v + delta2:
                    r_l = True

        # Random lan changing with probability pc. 随机变道
        if safe_r is True and l_r is True and random.random() < pc:
            # print("l_to_r")
            return "l_to_r", front_index_r

        elif safe_l is True and r_l is True and random.random() <= pc:
            # print("r_to_l")
            return "r_to_l", front_index_l

        else:
            front_index = car_index-1
            return "no", front_index

    def lan_changing(self, safe_d=1, pc=0.07, gc=3, delta1=1, delta2=1):
        """Run lane changing for the highway."""
        new_carlist = copy.deepcopy(self.carlist)
        new_anglelist = copy.deepcopy(self.anglelist)

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
        # print("Here", len(ch_ls[1]), len(self.carlist[1]))

        for lane in range(len(ch_ls)):  # each lane'
            # print("Here22", len(ch_ls[1]), len(self.carlist[1]))
            minus = 0
            for i, m in enumerate(ch_ls[lane]):  # each car
                # print(lane, i, len(ch_ls[lane]), len(self.carlist[lane]))
                o_lane = lane
                new_lane = lane
                front_index = f_index_ls[lane][i]
                if m == "l_to_r":  # change to the right lane, 向右变道
                    new_lane = lane + 1
                elif m == "r_to_l":  # change to the left lane, 向左变道
                    new_lane = lane - 1
                elif m == "no":  # Not change, 不变道
                    pass
                else:
                    raise NameError('Error for lan-changing!')

                # print(i, len(self.carlist[o_lane]))

                # update the car list on the highway
                if m != "no" and front_index is not None:  # if lane changing
                    # print(m, lane, new_lane, len(new_carlist))
                    # print(m, len(new_carlist[new_lane]), front_index+1)
                    # print(i, len(self.carlist[o_lane]))
                    l = new_carlist[o_lane].pop(i-minus)
                    a = new_anglelist[o_lane].pop(i-minus)
                    minus += 1
                    # print(front_index)
                    new_carlist[new_lane].insert(front_index+1, l)
                    new_anglelist[new_lane].insert(front_index+1, a)

                    # new_carlist[new_lane][front_index+2:] = self.carlist[new_lane][front_index+1:]
                    # new_anglelist[new_lane][front_index+2:] = self.anglelist[new_lane][front_index+1:]

                    # new_carlist[o_lane].pop(i)
                    # new_anglelist[o_lane].pop(i)
                # print("Here", len(ch_ls[1]), len(self.carlist[1]))

        self.carlist = new_carlist
        self.anglelist = new_anglelist

    def update(self, safe_d=1, pc=0.07, p=0.5, gc=3, delta1=1, delta2=1, k=3):
        """Update the position of the car.
        pc = the probability of overtaking/lane-changing.
        p = the probability of velocity slowdown.
        gc = lane change safety distance
        G = k*v = the safe distance, set G=v """
        # self.lane = lane
        self.lan_changing(safe_d=safe_d, pc=pc, gc=gc, delta1=delta1, delta2=delta2)

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
                # print(car.v, g)
                if car.v >= 0 and g >= 0:
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

    def density_1(self):
        """Calculate the car density.
        tot_car/RL."""
        density = self.tot_cars/self.road_length
        # print(self.tot_cars, self.road_length, density)
        return density

    def density_2(self):
        """Calculate the car density.
        sum(tot_car_lane/RL)/num_lane."""
        sum_density = 0
        for lane in self.carlist:
            sum_density += len(lane)
        density = sum_density/self.num_lane
        return density

    def flow_rate_1(self):
        """Calculate the flow rate of the highway.
        sum_v/RL. """
        tot_v = 0
        for lane in self.carlist:
            for car in lane:
                tot_v += car.v
        flow = tot_v/self.road_length
        return flow

    def flow_rate_2(self):
        """Calculate the flow rate of the highway.
        sum(sum_v_lane/RL)/num_lane ."""
        tot_flow = 0
        for lane in self.carlist:
            tot_v_lane = 0
            for car in lane:
                tot_v_lane += car.v
            tot_flow += tot_v_lane/self.road_length
        flow_mean = tot_flow/self.num_lane
        return flow_mean


def animate(frameNr, ax, road, p, pc):
    ax.clear()
    ax.set_rlim(0, 1.5)
    ax.axis('off')
    road.update(p=p, pc=pc)
    new_r = [[] for i in range(len(road.anglelist))]
    for lane in range(len(road.anglelist)):
        for i in range(len(road.anglelist[lane])):
            new_r[lane].append(lane+1)
    for lane in range(len(road.anglelist)):
        print(len(road.anglelist[lane]), len(new_r[lane]))
        ax.scatter(road.anglelist[lane], new_r[lane])
    return


class Simulation:
    """Simulation traffic."""
    def __init__(self, road, pc, p, mark="b"):
        self.road =road
        self.p = p
        self.pc = pc
        self.mark = mark

    def run(self, num_lane, tot_cars=10, max_v=2, p=0.5, pc=0.07):
        self.road = Road(max_v=max_v, road_length=50, max_car=tot_cars, num_lane=num_lane)
        for lane in range(num_lane):
            for i in range(tot_cars):
                self.road.add_car(Car(), lane)

        posls_o = [[], []]
        for lane in range(2):
            for c in self.road.carlist[lane]:
                posls_o[lane].append(c.pos)
        # print(posls_o)

        for i in range(100):
            self.road.update(pc=pc, p=p)

        posls = [[], []]
        for lane in range(2):
            for c in self.road.carlist[lane]:
                posls[lane].append(c.pos)
        # print(posls)

    def flowrate_dens_mean(self, num_lane, num_carlist, tot_cars=10, max_v=2, p=0.01, pc=0.07, t_start=1000, time=100, gc=3, delta1=1, delta2=1,k=3,safe_d=1):
        """Plot the flow rate vs the different densities."""
        flow_rate = []
        density = []

        for i in num_carlist:
            self.road = Road(max_v=max_v, road_length=i*tot_cars, max_car=tot_cars, num_lane=num_lane)
            flow = 0

            for j in range(tot_cars):
                for lane in range(num_lane):
                    self.road.add_car(Car(), lane, "b")

            # Flow rate is calculated when the system is balance that is after t_star steps.
            for j in range(t_start):
                self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)

            # Now begin to note the flow rate
            for j in range(time):
                flow += self.road.flow_rate_2()
                self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)

            flow = flow/time
            flow_rate.append(flow)
            density.append(self.road.density_1())

        plt.plot(density, flow_rate, '.--')
        plt.title('Fundamental diagram, max_v = '+str(max_v) + ', pc =' + str(pc) + ', p =' + str(p) + ', num-lane = ' +
                  str(num_lane))
        plt.xlabel('Density')
        plt.ylabel('Flow rate')
        plt.show()

    def plot_flowrate_time(self, repeated=1000, road_length=50, num_lane=2, max_v=2, tot_cars=10, p=0.5, pc=0.07, end_time=1000):
        """Plot the flow rate vs time."""
        flowrate = [0]*(end_time+1)
        t = []
        for i in range(end_time+1):
            t.append(i)

        for i in range(repeated):
            self.road = Road(road_length=road_length, max_v=max_v, max_car=tot_cars,num_lane=num_lane)
            for j in range(tot_cars):
                for lane in range(num_lane):
                    self.road.add_car(Car(), lane, "b")

            for j in range(end_time):
                flowrate[j] += self.road.flow_rate_1()
                self.road.update(pc=pc, p=p)
            flowrate[end_time] += self.road.flow_rate_1()

        for i in range(len(flowrate)):
            flowrate[i] = flowrate[i]/repeated

        plt.plot(t, flowrate)
        plt.title("Flow rate vs time, , roadLength = "+str(road_length) + " max_v = "+str(max_v)+" pc = "+str(pc) + " p = " +str(p))
        plt.xlabel('Time')
        plt.ylabel('Flow rate')
        plt.show()

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
    pc = 0.07
    # gc=3
    delta1=1
    delta2=1
    k=3
    safe_d=1
    road_length = 50
    max_v = 2
    num_lane = 2
    road = Road(max_v=max_v, road_length=road_length, max_car=25, num_lane=num_lane)
    simulation = Simulation(road=road, p=p, pc=pc, mark="b")
    # simulation.run(num_lane=num_lane, tot_cars=10, max_v=max_v, p=p, pc=pc)
    # simulation.run_animate(tot_car=25, mark="b", time=100, stepsperframe=1, title="Simulation")
    numcarlist = [i for i in range(1, 11)]
    more = [i * 10 for i in range(1, 51)]
    numcarlist.extend(more)
    simulation.flowrate_dens_mean(num_lane=num_lane, num_carlist=numcarlist, max_v=max_v, p=p, pc=pc, delta1=delta1, delta2=delta2, k=k, safe_d=safe_d)
    # simulation.plot_flowrate_time(road_length=50, num_lane=2, max_v=2, tot_cars=10, p=0.5, pc=0.07, end_time=1000)


if __name__ == '__main__':
    main()






