#!/usr/bin/env python3
import threading
from ev3dev2.sensor.lego import ColorSensor
from ev3dev2.sensor.lego import UltrasonicSensor
# from ev3dev2.sensor import Sensor
import time
import copy
import pickle
from sensor_distributions import mean_and_var


class Sensors:

    def __init__(self, tank, max_data=120, discrete=False):
        # Discrete
        self.discrete = discrete  # If discrete do not calculate gps location data

        # Thread lock
        self.event = threading.Event()
        self.semaphore = threading.Semaphore()
        self.rate = 0.01  # Sleep time between sensor readings

        # Data logs
        self.max_data = max_data
        self.data_log = {}
        self.counter = 0

        # Actual sensor objects
        tank.reset()
        self.color_sensor = ColorSensor()
        self.color_sensor_offset = 2.5
        self.ultrasonic_sensor = UltrasonicSensor()
        self.left_wheel = tank.left_motor
        self.right_wheel = tank.right_motor
        self.left_wheel.reset()
        self.right_wheel.reset()
        # self.imu = Sensor('ev3-  # RIP IMU
        # self.imu.mode = 'ACCEL'

        # Sensor data
        self.sensor_data = {"color": {"data": 0, "time": time.time(), "count": 0, "gps": -10},
                            "ultrasonic": {"data": 0, "time": time.time()},
                            "left": {"data": 0, "time": time.time()},
                            #"imu": {"data": 0, "time": time.time()},
                            "right": {"data": 0, "time": time.time()}}

        # Initialize all data to zero, predefining in dictionary hopefully speeds things up
        for i in range(self.max_data):
            self.data_log[i] = {"color": {"data": 0, "time": 0, "count": 0, "gps": 0},
                                "ultrasonic": {"data": 0, "time": 0},
                                "left": {"data": 0, "time": 0},
                                # "imu": {"data": 0, "time": 0},
                                "right": {"data": 0, "time": 0}}

        # Get gps points from sensor_distributions.py
        self.gps_points = (-1,
                           mean_and_var['gps'][1]['mean'],
                           mean_and_var['gps'][2]['mean'],
                           mean_and_var['gps'][3]['mean'],
                           -1,
                           -1,
                           -1)

        # Sensor threads
        self.threads = [threading.Thread(target=self.color_thread),
                        threading.Thread(target=self.wheel_thread, args=('l',)),
                        threading.Thread(target=self.wheel_thread, args=('r',)),
                        #threading.Thread(target=self.imu_thread),
                        threading.Thread(target=self.ultrasonic_thread)]
        self.num_threads = len(self.threads)
        self.thread_count = 0

        # Start threads
        for thread in self.threads:
            thread.start()

    def synchronise(self):
        """Called by each thread to ensure sensors are synced"""
        with self.semaphore:
            self.thread_count += 1
            if self.thread_count == self.num_threads:
                self.thread_count = 0
                self.event.set()
        self.event.wait()

    def color_thread(self):
        """Color sensor data is a string of color, if not discrete will add GPS mean and counter"""
        t = threading.currentThread()
        self.synchronise()

        while getattr(t, "do_run", True):
            prev_color = self.sensor_data['color']['data']
            self.sensor_data['color']['data'] = self.color_sensor.color_name
            # If not discrete and color sensor is Red update red counts and get mean gps reading for that count
            if self.discrete is False:
                if self.sensor_data['color']['data'] == 'Red':
                    if prev_color != 'Red':
                        self.sensor_data['color']['count'] += 1
                    self.sensor_data['color']['gps'] = self.gps_points[self.sensor_data['color']['count']]
                else:
                    self.sensor_data['color']['gps'] = -10
            self.sensor_data['color']['time'] = time.time()
            time.sleep(self.rate)
            self.synchronise()

    def imu_thread(self):
        """IMU sensor data is the x acceleration"""
        t = threading.currentThread()
        self.synchronise()

        while getattr(t, "do_run", True):
            accel = self.imu.value(0)
            self.sensor_data['imu']['data'] = accel
            self.sensor_data['imu']['time'] = time.time()
            time.sleep(self.rate)
            self.synchronise()

    def ultrasonic_thread(self):
        """Ultrasonic sensor measures distance to wall, returns difference from start point to current point if result
        is negative it will simply return the previous result"""
        t = threading.currentThread()
        self.synchronise()

        while getattr(t, "do_run", True):
            distance = 220 - self.ultrasonic_sensor.distance_centimeters
            if distance > 0:
                self.sensor_data['ultrasonic']['data'] = distance
            self.sensor_data['ultrasonic']['time'] = time.time()
            time.sleep(self.rate)
            self.synchronise()

    def wheel_thread(self, wheel):
        """Wheel thread data is the position of the wheel encoder for either left or right"""
        t = threading.currentThread()
        self.synchronise()

        if wheel == 'l':
            while getattr(t, "do_run", True):
                self.sensor_data['left']['data'] = self.left_wheel.position
                self.sensor_data['left']['time'] = time.time()
                time.sleep(self.rate)
                self.synchronise()

        elif wheel == 'r':
            while getattr(t, "do_run", True):
                self.sensor_data['right']['data'] = self.right_wheel.position
                self.sensor_data['right']['time'] = time.time()
                time.sleep(self.rate)
                self.synchronise()

    def get_sensor_data_copy(self):
        """Gets all sensor data ensuring, deep copy required to prevent creating a pointer to current object"""
        data = copy.deepcopy(self.sensor_data)
        self.update_data_log(data)
        return data

    def end_threads(self):
        """Changes do_run variable to false and ends all threads"""
        for thread in self.threads:
            thread.do_run = False

    def update_data_log(self, data):
        """Writes data to data log"""
        self.data_log[self.counter] = data
        self.counter += 1

    def write_data_log(self):
        """Saves data log to pickle file, removes unused data dictionaries"""
        for i in range(self.counter, self.max_data):
            del self.data_log[i]
        filename = 'data_log.pickle'
        with open(filename, 'wb') as fp:
            pickle.dump(self.data_log, fp)
        print("Pickled {}".format(filename))


def average_timestep(all_data, calculate_sync=False):
    """Takes in one chunk of data from sensors and finds the average timestep across all time stamps"""
    average_time = 0.0
    for i in all_data:
        average_time += all_data[i]['time']
    average_time = average_time / len(all_data)
    if calculate_sync:
        """Determine the average distance from the mean"""
        error = 0
        for i in all_data:
            error += abs(all_data[i]['time'] - average_time)
        return error/4
    else:
        return average_time


def adjust_timesteps(all_data, delta_time):
    """Adds time to all timestamps"""
    current = time.time()
    for i in all_data:
        all_data[i]['time'] = current + delta_time
    return all_data
