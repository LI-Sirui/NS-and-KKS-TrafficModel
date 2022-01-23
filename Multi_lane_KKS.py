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
        # self.angle = angle  # This is used to a circular road
        # self.r = lane + 1 # radius of the circular road


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

        # pick
        # self.num_car = 0  # the number of the cars on the lane, pick, change always
        # self.lane = 0  # pick of the lane change always

    def add_car(self, car, lane):
        """Add car to the lane.
        There are two kind of the initial position.
        if mark = b: The cars line up at the starting point of the road.
    """

        if len(self.carlist[lane]) < self.max_cars:
            self.tot_cars += 1
            # The cars line up at the starting point of the road.
            car.pos = (self.road_length - len(self.carlist[lane])) % self.road_length
            car.angle = car.pos * 2 * np.pi / self.road_length
            self.carlist[lane].append(car)

            # for c in self.carlist[0]:
                # print(c.pos)

    def find_car_index_leftright(self, carlist, car, lane):
        """To find the index of the front car and back car on the right, left lane."""
        mark1 = False
        mark2 = False
        back_index_r = 0
        back_index_l = 0
        front_index_l = 0
        front_index_r = 0
        # print(lane, len(carlist))
        # Find the index of car on the right lane.
        if lane + 1 < self.num_lane:  # There is a lane on the right
            # print("hi", lane, self.num_lane)
            mark1 = True
            if len(carlist[lane + 1]) > 0:  # There are least 2 car on the right lane.
                max_index = 0
                min_index = 0
                for cc_index, cc in enumerate(carlist[lane+1]):
                    if cc.pos > carlist[lane+1][max_index].pos:
                        max_index = cc_index
                    if cc.pos < carlist[lane+1][min_index].pos:
                        min_index = cc_index

                if car.pos > carlist[lane+1][max_index].pos or car.pos <= carlist[lane+1][min_index].pos:
                    back_index_r = max_index
                    front_index_r = min_index
                else:
                    max_index = 0
                    for cc_index, cc in enumerate(carlist[lane + 1]):
                        if cc.pos > carlist[lane + 1][max_index].pos and cc.pos <= car.pos:
                            max_index = cc_index
                    back_index_r = max_index
                    front_index_r = max_index-1
                    # if front_index_r == len(self.carlist[lane+1]):
                    #     front_index_r = 0

            elif len(carlist[lane + 1]) == 0:
                back_index_r, front_index_r = None, None

        # To find the index of car on the left lane.
        if lane > 0:  # There is a lane on the left.
            mark2 = True
            if len(carlist[lane - 1]) > 0:  # There is least two cars on the left lane.
                max_index = 0
                min_index = 0
                for cc_index, cc in enumerate(carlist[lane - 1]):
                    if cc.pos > carlist[lane - 1][max_index].pos:
                        max_index = cc_index
                    if cc.pos < carlist[lane - 1][min_index].pos:
                        min_index = cc_index

                if car.pos > carlist[lane - 1][max_index].pos or car.pos <= carlist[lane - 1][min_index].pos:
                    back_index_l = max_index
                    front_index_l = min_index
                else:
                    max_index = 0
                    for cc_index, cc in enumerate(carlist[lane - 1]):
                        if cc.pos > carlist[lane - 1][max_index].pos and cc.pos <= car.pos:
                            max_index = cc_index
                    back_index_l = max_index
                    front_index_l = max_index-1
                    # if front_index_l == len(self.carlist[lane-1]):
                    #     front_index_l = 0

            elif len(carlist[lane - 1]) == 0:
                back_index_l, front_index_l = None, None

        # print("here",lane)

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
        # print("111", lane)
        mark1, mark2, front_index_r, back_index_r, front_index_l, back_index_l = self.find_car_index_leftright(carlist,
                                                                                                               car, lane)
        # print("222", lane)
        # Calculate the distance between cars
        if mark1 is True and front_index_r is not None and back_index_r is not None:
            g_front_r = (carlist[lane + 1][front_index_r].pos - car.pos - safe_d + self.road_length) % self.road_length
            g_back_r = (car.pos - carlist[lane + 1][back_index_r].pos - safe_d + self.road_length) % self.road_length
            # print("right", front_index_r,back_index_r, g_front_r, g_back_r)

            posls = [[], [], []]
            for ett_lane in range(2):
                for c in self.carlist[ett_lane]:
                     posls[ett_lane].append(c.pos)
            # print("pos", posls)
            # print("Car", car.pos)

        if mark2 is True and front_index_l is not None and back_index_l is not None:
            g_front_l = (carlist[lane - 1][front_index_l].pos - car.pos - safe_d + self.road_length) % self.road_length
            g_back_l = (car.pos - carlist[lane - 1][back_index_l].pos - safe_d + self.road_length) % self.road_length

        # print(g_front_l, g_back_l, g_front_r, g_back_l)

        # Rule 1, Safety Conditions
        if mark1 is True:
            # print(back_index_r, len(carlist), lane)
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
            # print("l_to_r", front_index_r)
            return "l_to_r", front_index_r

        elif safe_l is True and r_l is True and random.random() <= pc:
            # print("r_to_l", front_index_l)
            return "r_to_l", front_index_l

        else:
            front_index = car_index-1
            # print("no")
            return "no", front_index

    def sim_lan_changing(self, carlist, car, lane, safe_d=1, pc=0.07, gc=3, delta1=1, delta2=1):
        car_index = None
        for i in range(len(carlist[lane])):
            if car.pos == carlist[lane][i].pos:
                car_index = i

        if car_index != None:
            new_carlist = copy.deepcopy(carlist)
            changing, front_index = self.infor_len_changing(carlist=new_carlist, car_index=car_index, car=car, lane=lane,
                                                                        safe_d=safe_d, pc=pc, gc=gc, delta1=delta1,
                                                                        delta2=delta2)

            posls = [[], [], []]
            for x in range(2):
                for c in carlist[x]:
                    posls[x].append(c.pos)
            # print("pos", posls)
            # print(car_index, changing, front_index)

            o_lane = lane
            new_lane = lane
            if changing == "l_to_r":  # change to the right lane, 向右变道
                new_lane = lane + 1
            elif changing == "r_to_l":  # change to the left lane, 向左变道
                new_lane = lane - 1
            elif changing == "no":  # Not change, 不变道
                pass
            else:
                raise NameError('Error for lan-changing!')

            if changing != "no":  # if lane changing
                l = new_carlist[o_lane].pop(car_index)
                if front_index is not None:
                    new_carlist[new_lane].insert(front_index + 1, l)
                else:
                    new_carlist[new_lane].append(l)

            return new_carlist

    def all_lan_changing(self, safe_d=1, pc=0.07, gc=3, delta1=1, delta2=1):
        posls = [[], [], []]
        for ett_lane in range(2):
            for c in self.carlist[ett_lane]:
                posls[ett_lane].append(c.pos)
        # print("pos", posls)
        carlist = copy.deepcopy(self.carlist)
        # print(len(self.carlist))
        for lane in range(len(self.carlist)):
            # print(lane)
            for car_index, car in enumerate(self.carlist[lane]):
                carlist = self.sim_lan_changing(carlist=carlist, car=car, lane=lane, safe_d=1, pc=0.07, gc=3, delta1=1, delta2=1)
                # print(carlist)
                # print(carlist[lane])

        # self.carlist = carlist

    def lan_changing(self, safe_d=1, pc=0.07, gc=3, delta1=1, delta2=1):
        """Run lane changing for the highway."""
        posls = [[], [], []]
        for ett_lane in range(2):
            for c in self.carlist[ett_lane]:
                posls[ett_lane].append(c.pos)
        # print("pos", posls)

        new_carlist = copy.deepcopy(self.carlist)

        ch_ls = [[] for j in range(self.num_lane)]
        f_index_ls = [[] for j in range(self.num_lane)]

        for lane, lanels in enumerate(self.carlist):
            for i, car in enumerate(self.carlist[lane]):
                changing, front_index = self.infor_len_changing(carlist=new_carlist, car_index=i, car=car, lane=lane,
                                                                safe_d=safe_d, pc=pc, gc=gc, delta1=delta1,
                                                                delta2=delta2)
                ch_ls[lane].append(changing)
                f_index_ls[lane].append(front_index)

        # new_f_index_ls = f_index_ls[:]
        # print("Here", len(ch_ls[1]), len(self.carlist[1]))
        # print("front", ch_ls)
        # print("index", f_index_ls)
        # posls = [[], [], []]
        # for lane in range(3):
        #     for c in self.carlist[lane]:
        #         posls[lane].append(c.pos)
        # print("pos", posls)

        for lane in range(len(ch_ls)):  # each lane'
            # print("Here22", len(ch_ls[1]), len(self.carlist[1]))
            minus = 0
            add = 0
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
                if m != "no":  # if lane changing
                    # print(m, lane, new_lane, len(new_carlist))
                    # print(m, len(new_carlist[new_lane]), front_index+1)
                    # print(i, len(self.carlist[o_lane]))
                    l = new_carlist[o_lane].pop(i-minus)
                    minus += 1
                    # print(front_index)
                    if front_index is not None:
                        new_carlist[new_lane].insert(front_index+1+add, l)
                        add += 1
                    else:
                        new_carlist[new_lane].append(l)
                        add += 1



                    # new_carlist[new_lane][front_index+2:] = self.carlist[new_lane][front_index+1:]
                    # new_anglelist[new_lane][front_index+2:] = self.anglelist[new_lane][front_index+1:]

                    # new_carlist[o_lane].pop(i)
                    # new_anglelist[o_lane].pop(i)
                # print("Here", len(ch_ls[1]), len(self.carlist[1]))

        self.carlist = new_carlist

    def update(self, safe_d=1, pc=0.07, p=0.5, gc=3, delta1=1, delta2=1, k=3):
        """Update the position of the car.
        pc = the probability of overtaking/lane-changing.
        p = the probability of velocity slowdown.
        gc = lane change safety distance
        G = k*v = the safe distance, set G=v """
        # self.lane = lane
        self.all_lan_changing(safe_d=safe_d, pc=pc, gc=gc, delta1=delta1, delta2=delta2)
        posls = [[], [], []]
        for lane in range(2):
            for c in self.carlist[lane]:
                posls[lane].append(c.pos)
        # print("pos", posls)

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
        vls = [[], [], []]
        for lane in range(2):
            for c in self.carlist[lane]:
                vls[lane].append(c.v)
        # print("v", vls)

        # update position of the cars
        self.update_pos()

    def update_pos(self):
        """Update position of the cars on the lane"""
        for lane_index, lane in enumerate(self.carlist):
            for i, car in enumerate(lane):
                car.pos = (car.pos + car.v) % self.road_length

    def density_1(self):
        """Calculate the car density.
        tot_car/(RL*num_lane)."""
        density = self.tot_cars/(self.road_length*self.num_lane)
        # print(self.tot_cars, self.road_length, density)
        return density

    """
    def flow_rate_1(self):
        tot_v = 0
        for lane in self.carlist:
            for car in lane:
                tot_v += car.v
        flow = tot_v/(self.road_length)
        return flow
    """

    def flow_rate_1(self):
        flowls = []
        for lane in self.carlist:
            tot_v_lane = 0
            for car in lane:
                tot_v_lane += car.v
            flowls.append(tot_v_lane/self.road_length)
        flow_max = max(flowls)
        # print(flowls)
        return flow_max


class Simulation:
    """Simulation traffic."""
    def __init__(self, road, pc, p):
        self.road =road
        self.p = p
        self.pc = pc

    def run(self, num_lane=2, tot_cars=50, max_v=2, p=0.5, pc=0.07):
        self.road = Road(max_v=max_v, road_length=50, max_car=tot_cars, num_lane=num_lane)
        # print(num_lane)
        if num_lane == 2:
            # print(num_lane)
            for i in range(tot_cars):
                pp = random.random()
                # print(pp)
                if pp <= 1/num_lane:
                    self.road.add_car(Car(), 0)
                elif 1/num_lane < pp <= 1:
                    self.road.add_car(Car(), 1)
        elif num_lane == 3:
            # print(num_lane)
            for i in range(tot_cars):
                pp = random.random()
                # print(pp)
                if pp <= 1/num_lane:
                    self.road.add_car(Car(), 0)
                elif 1/num_lane < pp <= 2/num_lane:
                    self.road.add_car(Car(), 1)
                elif 2/num_lane < pp <= 1:
                    self.road.add_car(Car(), 2)
        """
        for lane in range(num_lane):
            for j in range(tot_cars):
                self.road.add_car(Car(), lane)
        """
        #print(num_lane, self.road.num_lane, len(self.road.carlist[0]))
        v_o = [[], [], []]
        for lane in range(2):
            for c in self.road.carlist[lane]:
                v_o[lane].append(c.v)
        # print(v_o, self.road.tot_cars)

        for i in range(2):
            self.road.update(pc=pc, p=p)

        vls = [[], [], []]
        for lane in range(2):
            for c in self.road.carlist[lane]:
                vls[lane].append(c.v)
        # print(vls, self.road.tot_cars)

        # print(self.road.flow_rate_1())

        posls_o = [[], [], []]
        for lane in range(2):
            for c in self.road.carlist[lane]:
                posls_o[lane].append(c.pos)
        # print(posls_o)

        for i in range(100):
            self.road.update(pc=pc, p=p)

        posls = [[], [], []]
        for lane in range(2):
            for c in self.road.carlist[lane]:
                posls[lane].append(c.pos)
        # print(posls)

    def flowrate_dens_mean(self, num_carlist, num_lane, tot_cars=10, max_v=2, p=0.01, pc=0.07, t_start=1000, time=100, gc=3, delta1=1, delta2=1, k=3, safe_d=1, mark_flow="1"):
        """Plot the flow rate vs the different densities."""
        flow_rate = []
        density = []

        for i in num_carlist:
            self.road = Road(max_v=max_v, road_length=25, max_car=25, num_lane=num_lane)
            flow = 0
            if mark_flow == "1":
                if num_lane == 2:
                    for n in range(i):
                        pp = random.random()
                        if pp <= 1 / num_lane:
                            self.road.add_car(Car(), 0)
                        else:
                            self.road.add_car(Car(), 1)
                elif num_lane == 3:
                    for n in range(i):
                        pp = random.random()
                        if pp <= 1 / num_lane:
                            self.road.add_car(Car(), 0)
                        elif 1 / num_lane < pp <= 2 / num_lane:
                            self.road.add_car(Car(), 1)
                        else:
                            self.road.add_car(Car(), 2)
            else:
                for lane in range(num_lane):
                    for j in range(i):
                        self.road.add_car(Car(), lane)
                # print(self.road.max_cars, self.road.tot_cars)

            # Flow rate is calculated when the system is balance that is after t_star steps.
            for j in range(t_start):
                self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)

            # Now begin to note the flow rate
            for j in range(time):
                flow += self.road.flow_rate_1()
                self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)

            # print(self.road.tot_cars)
            flow = flow/time
            flow_rate.append(flow)
            # print(flow_rate)
            density.append(self.road.density_1())
        # print(density)
        # print(flow_rate)

        plt.plot(density, flow_rate, '.--', label="num-lane = " +str(num_lane))
        # plt.title('Fundamental diagram, max_v = '+str(max_v) + ', pc =' + str(pc) + ', p =' + str(p) + ', num-lane = ' +
        #           str(num_lane))
        # plt.xlabel('Density')
        # plt.ylabel('Flow rate')
        # plt.show()

    def plot_fund_lane(self, num_carlist, tot_cars=10, max_v=2, p=0.01, pc=0.07, t_start=1000, time=100, gc=3, delta1=1, delta2=1, k=3, safe_d=1, mark_flow="1"):
        """Fundamental diagram for different number of lanes."""
        for num_lane in range(2, 4):
            self.flowrate_dens_mean(num_lane=num_lane, num_carlist=num_carlist, max_v=max_v, p=p, pc=pc, delta1=delta1, delta2=delta2, k=k, safe_d=safe_d, mark_flow="1")
        plt.title(
            'Fundamental diagram, max_v = ' + str(max_v) + ', pc =' + str(pc) + ', p =' + str(p))
        plt.xlabel('Density')
        plt.ylabel('Flow rate')
        plt.legend()
        plt.show()

    def flow_stderror(self, num_lane, roadlength, tot_cars=25, max_v=2, pc=0.07, p=0.5, start_t=1000, time_flowmean=100, rep_sim=3, gc=3, delta1=1, delta2=1, k=3, safe_d=1, mark_flow="1"):
        """Calculate the standard error of the flow rate."""
        flowrate=[]
        std = []

        for i in range(rep_sim):
            print(i)
            self.road = Road(max_v=max_v, road_length=25, max_car=25, num_lane=num_lane)
            flow = 0
            if mark_flow == "1":
                if num_lane == 2:
                    for n in range(tot_cars):
                        pp = random.random()
                        if pp <= 1 / num_lane:
                            self.road.add_car(Car(), 0)
                        else:
                            self.road.add_car(Car(), 1)
                elif num_lane == 3:
                    for n in range(tot_cars):
                        pp = random.random()
                        if pp <= 1 / num_lane:
                            self.road.add_car(Car(), 0)
                        elif 1 / num_lane < pp <= 2 / num_lane:
                            self.road.add_car(Car(), 1)
                        else:
                            self.road.add_car(Car(), 2)
            else:
                for lane in range(num_lane):
                    for j in range(tot_cars):
                        self.road.add_car(Car(), lane)

            for j in range(start_t):
                self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)

            for j in range(time_flowmean):
                flow += self.road.flow_rate_1()
                self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)

            flow = flow /time_flowmean
            flowrate.append(flow)
            # print(flowrate)

            mean_sqrt = 0
            for flow in flowrate:
                mean_sqrt += flow**2
            mean_sqrt = mean_sqrt/len(flowrate)
            mean = sum(flowrate)/len(flowrate)

            if i > 1:
                std.append(np.sqrt((mean_sqrt - mean**2)/(i-1)))


        repnum_list = []
        # ref = []
        for i in range(2, rep_sim):
             repnum_list.append(i)
        #     ref.append(0.001)

        # plt.title('The standard error of the flow rate')
        # plt.xlabel('The repeated time of simulations')
        # plt.ylabel('The standard error')
        plt.plot(repnum_list, std, label='The standard error, num-lane = ' + str(num_lane))
        # plt.plot(repnum_list, ref, label='Reference 0.001')
        # plt.legend()
        # plt.show()

    def more_lane_std(self, roadlength, tot_cars=25, max_v=2, pc=0.07, p=0.5, start_t=1000, time_flowmean=100, rep_sim=3, gc=3, delta1=1, delta2=1, k=3, safe_d=1, mark_flow="1"):
        """Plot the standard error of the flow rate with 2-lanes and 3-lanes at the same time."""
        for num_lane in range(2, 4):
            self.flow_stderror(num_lane=num_lane, roadlength=25, tot_cars=25, rep_sim=100)

        # repnum_list = []
        ref = []
        for i in range(2, rep_sim):
            # repnum_list.append(i)
            ref.append(0.001)

        plt.title('The standard error of the flow rate, RoadLength = ' + str(25) + 'tot-cars = ' +str(25))
        plt.xlabel('The repeated time of simulations')
        plt.ylabel('The standard error')
        plt.plot(repnum_list, ref, label='Reference 0.001')
        plt.legend()
        plt.show()

    def plot_flowrate_time(self, num_lane, road_length, tot_cars=25, max_v=2, pc=0.07, p=0.5, repeated=100, end_time=500
                           , gc=3, delta1=1, delta2=1, k=3, safe_d=1, mark_flow="1"):
        """Plot the flow rate vs time."""
        flowrate = [0]*(end_time+1)
        t = []

        for i in range(end_time+1):
            t.append(i)

        for i in range(repeated):
            print(i)
            self.road = Road(max_v=max_v, road_length=25, max_car=25, num_lane=num_lane)
            flow = 0
            if mark_flow == "1":
                if num_lane == 2:
                    for n in range(tot_cars):
                        pp = random.random()
                        if pp <= 1 / num_lane:
                            self.road.add_car(Car(), 0)
                        else:
                            self.road.add_car(Car(), 1)
                elif num_lane == 3:
                    for n in range(tot_cars):
                        pp = random.random()
                        if pp <= 1 / num_lane:
                            self.road.add_car(Car(), 0)
                        elif 1 / num_lane < pp <= 2 / num_lane:
                            self.road.add_car(Car(), 1)
                        else:
                            self.road.add_car(Car(), 2)
            else:
                for lane in range(num_lane):
                    for j in range(tot_cars):
                        self.road.add_car(Car(), lane)

            for j in range(end_time):
                flowrate[j] += self.road.flow_rate_1()
                self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)
            flowrate[end_time] += self.road.flow_rate_1()

        for i in range(len(flowrate)):
            flowrate[i] = flowrate[i]/repeated

        plt.plot(t, flowrate)
        plt.title("Flow rate vs time, , roadLength = "+str(road_length) + " max_v = "+str(max_v)+" pc = "+str(pc) + " p = " +str(p))
        plt.xlabel('Time')
        plt.ylabel('Flow rate')
        plt.show()

    def diff_v_flowVsdensity(self, num_carlist, num_lane, tot_cars=10, max_v=2, p=0.01, pc=0.07, t_start=1000, time=100, gc=3, delta1=1, delta2=1, k=3, safe_d=1, mark_flow="1"):
        """Plot the fundamental diagram for different velocities."""
        vlist = [1, 2, 5]
        for v in vlist:
            flow_rate = []
            density = []
            for i in num_carlist:
                self.road = Road(max_v=v, road_length=25, max_car=25, num_lane=num_lane)
                flow = 0
                if mark_flow == "1":
                    if num_lane == 2:
                        for n in range(i):
                            pp = random.random()
                            if pp <= 1 / num_lane:
                                self.road.add_car(Car(), 0)
                            else:
                                self.road.add_car(Car(), 1)
                    elif num_lane == 3:
                        for n in range(i):
                            pp = random.random()
                            if pp <= 1 / num_lane:
                                self.road.add_car(Car(), 0)
                            elif 1 / num_lane < pp <= 2 / num_lane:
                                self.road.add_car(Car(), 1)
                            else:
                                self.road.add_car(Car(), 2)
                else:
                    for lane in range(num_lane):
                        for j in range(i):
                            self.road.add_car(Car(), lane)
                    # print(self.road.max_cars, self.road.tot_cars)

                # Flow rate is calculated when the system is balance that is after t_star steps.
                for j in range(t_start):
                    self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)

                # Now begin to note the flow rate
                for j in range(time):
                    flow += self.road.flow_rate_1()
                    self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)

                # print(self.road.tot_cars)
                flow = flow/time
                flow_rate.append(flow)
                # print(flow_rate)
                density.append(self.road.density_1())
            # print(density)
            # print(flow_rate)

            plt.plot(density, flow_rate, '.--', label="v = " + str(v) + "num-lane = " +str(num_lane))
        plt.title('Fundamental diagram with different max_v, pc =' + str(pc) + ', p =' + str(p))
        plt.xlabel('Density')
        plt.ylabel('Flow rate')
        plt.show()

    def diff_p_flowVSdenstiy(self, num_carlist, num_lane, tot_cars=10, max_v=2, pc=0.07, t_start=1000, time=100, gc=3, delta1=1, delta2=1, k=3, safe_d=1, mark_flow="1"):
        """Plot the fundamental diagram for different velocities."""
        plist = [0.2, 0.5, 0.8]
        for p in plist:
            flow_rate = []
            density = []
            for i in num_carlist:
                self.road = Road(max_v=v, road_length=25, max_car=25, num_lane=num_lane)
                flow = 0
                if mark_flow == "1":
                    if num_lane == 2:
                        for n in range(i):
                            pp = random.random()
                            if pp <= 1 / num_lane:
                                self.road.add_car(Car(), 0)
                            else:
                                self.road.add_car(Car(), 1)
                    elif num_lane == 3:
                        for n in range(i):
                            pp = random.random()
                            if pp <= 1 / num_lane:
                                self.road.add_car(Car(), 0)
                            elif 1 / num_lane < pp <= 2 / num_lane:
                                self.road.add_car(Car(), 1)
                            else:
                                self.road.add_car(Car(), 2)
                else:
                    for lane in range(num_lane):
                        for j in range(i):
                            self.road.add_car(Car(), lane)
                    # print(self.road.max_cars, self.road.tot_cars)

                # Flow rate is calculated when the system is balance that is after t_star steps.
                for j in range(t_start):
                    self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)

                # Now begin to note the flow rate
                for j in range(time):
                    flow += self.road.flow_rate_1()
                    self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)

                # print(self.road.tot_cars)
                flow = flow/time
                flow_rate.append(flow)
                # print(flow_rate)
                density.append(self.road.density_1())
            # print(density)
            # print(flow_rate)

            plt.plot(density, flow_rate, '.--', label="p = " + str(p) + "num-lane = " + str(num_lane))
        plt.title('Fundamental diagram with different p, max_v = '+str(max_v) + ', pc =' + str(pc))
        plt.xlabel('Density')
        plt.ylabel('Flow rate')
        plt.show()

    def diff_pc_flowVSdenstiy(self, num_carlist, num_lane, tot_cars=10, max_v=2, pc=0.07, t_start=1000, time=100, gc=3, delta1=1, delta2=1, k=3, safe_d=1, mark_flow="1"):
        """Plot the fundamental diagram for different velocities."""
        pclist = [0.07, 0.4, 0.8]
        for pc in pclist:
            flow_rate = []
            density = []
            for i in num_carlist:
                self.road = Road(max_v=v, road_length=25, max_car=25, num_lane=num_lane)
                flow = 0
                if mark_flow == "1":
                    if num_lane == 2:
                        for n in range(i):
                            pp = random.random()
                            if pp <= 1 / num_lane:
                                self.road.add_car(Car(), 0)
                            else:
                                self.road.add_car(Car(), 1)
                    elif num_lane == 3:
                        for n in range(i):
                            pp = random.random()
                            if pp <= 1 / num_lane:
                                self.road.add_car(Car(), 0)
                            elif 1 / num_lane < pp <= 2 / num_lane:
                                self.road.add_car(Car(), 1)
                            else:
                                self.road.add_car(Car(), 2)
                else:
                    for lane in range(num_lane):
                        for j in range(i):
                            self.road.add_car(Car(), lane)
                    # print(self.road.max_cars, self.road.tot_cars)

                # Flow rate is calculated when the system is balance that is after t_star steps.
                for j in range(t_start):
                    self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)

                # Now begin to note the flow rate
                for j in range(time):
                    flow += self.road.flow_rate_1()
                    self.road.update(safe_d=safe_d, pc=pc, p=p, gc=gc, delta1=delta1, delta2=delta2, k=k)

                # print(self.road.tot_cars)
                flow = flow/time
                flow_rate.append(flow)
                # print(flow_rate)
                density.append(self.road.density_1())
            # print(density)
            # print(flow_rate)

            plt.plot(density, flow_rate, '.--', label="p = " + str(p) + "num-lane = " + str(num_lane))
        plt.title('Fundamental diagram with different p, max_v = '+str(max_v) + ', pc =' + str(pc))
        plt.xlabel('Density')
        plt.ylabel('Flow rate')
        plt.show()


def main():
    p = 0.5
    pc = 0.5
    # gc=3
    delta1=1
    delta2=1
    k=3
    safe_d=1
    road_length = 25
    max_v = 2
    num_lane = 2
    road = Road(max_v=max_v, road_length=road_length, max_car=25, num_lane=num_lane)
    simulation = Simulation(road=road, p=p, pc=pc)
    # simulation.run(num_lane=num_lane, tot_cars=10, max_v=max_v, p=p, pc=pc)
    # simulation.run_animate(tot_car=25, mark="b", time=100, stepsperframe=1, title="Simulation")
    # numcarlist = [i for i in range(0, 11)]
    # more = [i * 10 for i in range(1, 51)]
    # numcarlist.extend(more)
    numcarlist = [0, 1, 2, 5, 10, 15, 30, 45, 60, 75, 90, 100]
    # simulation.plot_fund_lane(num_carlist=numcarlist, max_v=max_v, p=p, pc=pc, delta1=delta1, delta2=delta2, k=k, safe_d=safe_d, mark_flow="1")
    # simulation.flowrate_dens_mean(num_lane=num_lane, num_carlist=numcarlist, max_v=max_v, p=p, pc=pc, delta1=delta1, delta2=delta2, k=k, safe_d=safe_d, mark_flow="1")
    # simulation.plot_flowrate_time(road_length=50, num_lane=2, max_v=2, tot_cars=25, p=p, pc=pc, end_time=1000)
    # simulation.flow_stderror(road_length=road_length, num_lane=2, tot_cars=25, rep_sim=250)
    simulation.diff_v_flowVsdensity(num_carlist=numcarlist, num_lane=num_lane, max_v=max_v, p=p, pc=pc, delta1=delta1, delta2=delta2, k=k, mark_flow="1")

if __name__ == '__main__':
    main()






