from ev3dev2.motor import MoveTank, OUTPUT_C, OUTPUT_D
from math import sqrt
from sensors import Sensors
import time
from missions import average_timestep
from odometry import Odometry

TIMESTEP = 0.25


def odom_noise(delta_time=4):
    """Turn motors on for a timestep, measure distance"""
    measurements = []
    errors = []
    tank = MoveTank(OUTPUT_C, OUTPUT_D)
    odometry = Odometry()
    s = Sensors(tank, discrete=True)

    while len(measurements) < 20:
        tank.on(15, 15)
        previous_state = s.get_sensor_data_copy()
        while time.time() - average_timestep(previous_state) < delta_time:
            pass
        tank.off()
        current_state = s.get_sensor_data_copy()
        odometry.calculate_velocity_and_distance(previous_state, current_state)
        actual_distance = float(input("how far in cm"))
        measurements.append(actual_distance)
        errors.append(abs(actual_distance-odometry.last_distance))

    mean, variance, std = get_mean_and_std(measurements, scale=delta_time/0.25)
    mean, variance, std = get_mean_and_std(errors, scale=delta_time / 0.25)


def gps_noise():
    """Move until you get a GPS reading, measure distance"""
    measurements = []
    tank = MoveTank(OUTPUT_C, OUTPUT_D)
    s = Sensors(tank, discrete=True)
    current_state = s.get_sensor_data_copy()
    while len(measurements) < 20:
        tank.on(15, 15)

        previous_state = current_state
        while time.time() - average_timestep(previous_state) < 0.25:
            pass
        current_state = s.get_sensor_data_copy()
        if current_state['color']['data'] == 'Red':
            tank.off()
            measurements.append(float(input("how far in cm")))

    mean, variance, std = get_mean_and_std(measurements)


def sonar_noise():
    """Travel till the sonar moves 1m, measure distance"""
    measurements = []
    tank = MoveTank(OUTPUT_C, OUTPUT_D)
    s = Sensors(tank, discrete=True)
    current_state = s.get_sensor_data_copy()
    while len(measurements) < 20:
        tank.on(15, 15)

        previous_state = current_state
        while time.time() - average_timestep(previous_state) < 0.25:
            pass
        current_state = s.get_sensor_data_copy()
        if current_state['sonar']['data'] >= 100:
            tank.off()
            measurements.append(float(input("how far in cm")))

    mean, variance, std = get_mean_and_std(measurements)


def get_mean_and_std(measurements, scale=1):
    mean = sum(measurements) / len(measurements)
    squared_diff = 0
    for measure in measurements:
        squared_diff += (measure - mean) ** 2
    variance = squared_diff / len(measurements)
    std = sqrt(variance/scale)
    print("Mean {} Variance {} STD {}".format(mean/scale, variance/scale, std))
    return mean, variance, std


if __name__ == '__main__':
    odom_noise()
