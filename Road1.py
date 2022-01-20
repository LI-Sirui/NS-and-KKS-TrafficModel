"""
Traffic model NS, a single lane highway.
The highway are assumed to be prototypes in order to observe animations.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import animation
from functools import reduce


class Car:
    """A simple car. 创建新的车辆，单一。"""
    r = 1  # radius of the circular road

    def __init__(self, pos=0, v=0, angle=0):
        self.pos = pos  # position
        self.v = v  # velocity
        self.angle = angle  # This is used to a circular raod


class Road:
    """A simple road."""
    def __init__(self, max_v=2, road_length=50, max_cars=25):

        # Road information
        self.road_length = road_length
        self.max_cars = max_cars

        # Road 1, the information of car.
        self.max_v = max_v
        self.tot_cars = 0
        self.carlist = []
        self.color = []

        self.anglelist = []  # used to a circular raod
        self.r = []  # used to a circular raod

    def add_car(self, car, mark):
        """Add car to the road. There are two kind of the initial position.
        if mark = a:  The cars are evenly distributed on the road.
        if mark = b: The cars line up at the starting point of the road.
        car = (pos, vel, angle)."""
        if len(self.carlist) < self.max_cars:
            self.tot_cars += 1

            # The initial positions of the cars are evenly distributed on the road. 汽车起初位置，平均分布在道路上
            if mark == "a":
                car.pos = (self.road_length - len(self.carlist) * self.road_length / self.max_cars) % self.road_length
                car.angle = len(self.carlist) * 2 * np.pi / self.max_cars

            # The cars line up at the starting point of the road.
            elif mark == "b":
                car.pos = (self.road_length - len(self.carlist)) % self.road_length
                car.angle = car.pos * 2 * np.pi / self.road_length

            self.color.append(len(self.carlist))
            self.carlist.append(car)
            self.anglelist.append(car.angle)
            self.r.append(car.r)

    def update(self, p):
        """ Uppdate the postion of the car. 更新汽车位置
        p =  the probability of velocity slowdown."""
        for i, car in enumerate(self.carlist):
            # Rule 1, acceleration， 加速
            if car.v < self.max_v:
                car.v += 1

            # Rule 2, hold the safetied distance d=5 between two car， 保持车距
            d = (self.carlist[i - 1].pos - car.pos + self.road_length) % self.road_length
            if car.v >= d != 0:
                car.v = d - 1

            # Rule 3, Random slowdown with probability p， 随机减速
            if car.v > 0 and random.random() < p:
                car.v -= 1

        # Rule 4 update position of the cars
        self.update_pos()

    def update_pos(self):
        """Update the positions of the cars on the road."""
        for i, car in enumerate(self.carlist):
            car.pos = (car.pos + car.v) % self.road_length
            self.anglelist[i] = car.pos * 2 * np.pi / self.road_length

    def density(self):
        """Calculate the car density. """
        density = self.tot_cars/self.road_length
        # print(self.tot_cars)
        return density

    def flow_rate(self):
        """Calculate the flow rate of the road.
        The flow rate = sum of the cars velocity / road length"""
        tot_v = 0
        for car in self.carlist:
            tot_v += car.v
        flow = tot_v/self.road_length
        return flow


def animate(frameNr, ax, road, p):
    ax.clear()
    ax.set_rlim(0, 1.5)
    ax.axis('off')
    road.update(p)
    return ax.scatter(road.anglelist, road.r, c=road.color),


class Simulation:
    """Simulation traffic."""
    def __init__(self, road, p, mark="b"):
        self.road = road
        self.p = p  # the probability of velocity slowdown
        self.mark = mark  # Initial distribution of cars

    def flowrate_dens_mean(self, numcarlist, tot_cars=10, max_v=2, p=0.5, t_start=1000, time=100):
        """Plot the flow rate vs the different densities."""
        flow_rate = []
        density = []

        for i in numcarlist:
            self.road = Road(road_length=i*tot_cars, max_v=max_v, max_cars=tot_cars)  # density decreases with road length increases
            flow = 0

            for j in range(tot_cars):
                self.road.add_car(Car(), mark=self.mark)

            # Flow rate is calculated when the system is balance that is after t_start steps.
            for j in range(t_start):
                self.road.update(p)

            # Now begin to note the flow rate.
            for j in range(time):
                flow += self.road.flow_rate()
                self.road.update(p)

            flow = flow/time  # the means of the flow rate
            flow_rate.append(flow)
            density.append(self.road.density())
        # print(density)
        plt.plot(density, flow_rate, '.--')
        plt.title("Fundamental diagram, max_v = "+str(max_v)+" p = "+str(p))
        plt.xlabel('density')
        plt.ylabel('Flow rate')
        plt.show()

    def plot_flowrate_time(self, repeated=1000, road_length=50, max_v=2, tot_cars=10, p=0.5, end_time=1000, mark="b"):
        """Plot the flow rate vs time."""
        flowrate=[0]*(end_time+1)
        t = []
        for i in range(end_time+1):
            t.append(i)

        for i in range(repeated):
            self.road = Road(road_length=road_length, max_v=max_v, max_cars=tot_cars)
            for j in range(tot_cars):
                self.road.add_car(Car(), mark=mark)
            for j in range(end_time):
                flowrate[j] += self.road.flow_rate()
                self.road.update(p)
            flowrate[end_time] += self.road.flow_rate()

        for i in range(len(flowrate)):
            flowrate[i] = flowrate[i]/repeated

        plt.plot(t, flowrate)
        plt.title("Flow rate vs time, roadLength = "+str(road_length) + " max_v = "+str(max_v)+" p = "+str(p))
        plt.xlabel('Time')
        plt.ylabel('Flow rate')
        plt.show()

    def flow_stderror(self, road_length=50, tot_cars=10, max_v=2, p=0.5, mark="a", start_t=1000, time_flowmean=100, rep_sim=100):
        """Calculate the standard error of the flow rate."""
        flowrate = []
        std = []
        for i in range(rep_sim):
            self.road = Road(road_length=road_length, max_v=max_v, max_cars=tot_cars)
            flow=0
            for j in range(tot_cars):
                self.road.add_car(Car(), mark=mark)
            for j in range(start_t):
                self.road.update(p)
            for j in range(time_flowmean):
                flow += self.road.flow_rate()
                self.road.update(p)
            flowrate.append(flow/time_flowmean)

            mean_sqrt = 0
            for flow in flowrate:
                mean_sqrt += flow**2
            mean_sqrt = mean_sqrt/len(flowrate)
            mean = sum(flowrate)/len(flowrate)

            if i > 1:
                std.append(np.sqrt((mean_sqrt-mean**2)/(i-1)))

        repnum_list = []
        ref = []
        for i in range(2, rep_sim):
            repnum_list.append(i)
            ref.append(0.001)

        plt.title("The standard error of the flow rate")
        plt.xlabel('The repeated time of simulations')
        plt.ylabel('The standard error')
        plt.plot(repnum_list, std, label='The standard error')
        plt.plot(repnum_list, ref, label='Reference 0.001')
        plt.legend()
        plt.show()

    def flowrate_differentRoadLength(self, maxroadL=400, max_v=2, start_t=1000, rep_meanflow=1000, p=0.5, mark="a"):
        """Plot the flow rate vs density which means different road length."""
        roadLength = 0
        while roadLength < maxroadL:
            if roadLength == 0:
                roadLength = 15
            else:
                roadLength = roadLength*2
            flowrate = []
            density = []
            tot_car = sorted(set(reduce(list.__add__,
                      ([j, roadLength // j] for j in range(1, int(roadLength ** 0.5) + 1) if roadLength % j == 0))))
            for n in tot_car:
                self.road = Road(road_length=roadLength, max_cars=n, max_v=max_v)
                flow=0
                for j in range(n):
                    self.road.add_car(Car(), mark=mark)
                for j in range(start_t):
                    self.road.update(p)
                for j in range(rep_meanflow):
                    flow += self.road.flow_rate()
                    self.road.update(p)
                flowrate.append(flow/rep_meanflow)
                density.append(self.road.density())
            plt.plot(density, flowrate, label="RoadLength = " +str(roadLength))
        plt.title("Fundamental diagram for different density, max_v = " + str(max_v) + " p = " + str(p))
        plt.xlabel("Density")
        plt.ylabel("Flow rate")
        plt.legend()
        plt.show()

    def diff_v_flowVSdensity(self, numcarlist, tot_cars=10, start_t=1000, rep_meanflow=1000, p=0.5, mark="a"):
        """Plot the fundamental diagram for different velocities."""

        vlist = [1, 2, 5]
        for v in vlist:
            flowrate = []
            density = []
            for i in numcarlist:
                self.road = Road(road_length=i * tot_cars, max_cars=tot_cars, max_v=v)
                flow = 0
                for j in range(tot_cars):
                    self.road.add_car(Car(), mark=mark)
                for j in range(start_t):
                    self.road.update(p)
                for j in range(rep_meanflow):
                    flow += self.road.flow_rate()
                    self.road.update(p)
                flowrate.append(flow/rep_meanflow)
                density.append(self.road.density())
            plt.plot(density, flowrate, label='max_v = ' + str(v))
        plt.title("The fundamental diagram with different velocity")
        plt.xlabel("Density")
        plt.ylabel("Flow rate")
        plt.legend()
        plt.show()

    def diff_p_flowVSdensity(self, numcarlist, tot_cars=10, start_t=1000, rep_flowmean=1000, mark="a"):
        """Plot the fundamental diagram for the simulations with different p"""
        plist = [0.2, 0.5, 0.8]
        for p in plist:
            flowrate = []
            density = []
            for i in numcarlist:
                self.road = Road(road_length=i * tot_cars, max_cars=tot_cars, max_v=2)
                flow = 0
                for j in range(tot_cars):
                    self.road.add_car(Car(), mark=mark)
                for j in range(start_t):
                    self.road.update(p)
                for j in range(rep_flowmean):
                    flow += self.road.flow_rate()
                    self.road.update(p)
                flowrate.append(flow/rep_flowmean)
                density.append(self.road.density())
            plt.plot(density, flowrate, label='p = ' + str(p))
        plt.title("The fundamental diagram with different p")
        plt.xlabel("Density")
        plt.ylabel("Flow rate")
        plt.legend()
        plt.show()

    def run_animate(self, tot_cars=25, mark="a", time=100, stepsperframe=1, title="Simulation", save=False):
        """Run the traffic animation."""
        for i in range(tot_cars):
            self.road.add_car(Car(), mark=mark)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.axis('off')
        ax.set_title(title, va='bottom')
        numframes = int(time / stepsperframe)
        anim = animation.FuncAnimation(fig, animate, fargs=[ax, self.road, self.p],
                                       frames=numframes, interval=50, blit=True, repeat=False, save_count=numframes)
        if save:
            writergif = animation.PillowWriter(fps=20)
            anim.save("car.gif", writer=writergif)
        else:
            plt.show()


def main():
    p = 0.50  # the probability of velocity slowdown.
    road_length = 50
    max_v = 2
    road = Road(road_length=road_length, max_cars=25, max_v=max_v)
    simulation = Simulation(road, p=p)
    simulation.run_animate(25, "b", 1000, 1, "Traffic simulator")
    # numcarlist = [i for i in range(1, 11)]
    # more = [i*10 for i in range(1, 51)]
    # numcarlist.extend(more)
    # simulation.flowrate_dens_mean(numcarlist)
    # simulation.plot_flowrate_time(road_length=road_length, tot_cars=25, max_v=max_v,p=p, end_time=200,mark="a")
    # simulation.flow_stderror(tot_cars=25, rep_sim=250)
    # simulation.flowrate_differentRoadLength()
    # simulation.diff_v_flowVSdensity(numcarlist=numcarlist, mark="a")
    # simulation.diff_p_flowVSdensity(numcarlist=numcarlist, mark="a")


if __name__ == '__main__':
    main()
