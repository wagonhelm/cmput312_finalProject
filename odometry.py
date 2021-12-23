from math import pi


class Odometry:
    """Methods for calculating distance and velocity based upon a wheels radius and revolutions"""

    def __init__(self):
        self.diameter = 5.55  # cm
        self.radius = self.diameter / 2
        self.velocity = 0
        self.total_distance = 0
        self.last_distance = 0

    def calculate_velocity_and_distance(self, x0: dict, x1: dict, DEBUG=False):
        """
        Takes in a previous state x0 and a current state x1, then calculates, the velocity and distance traveled between
        states in a straight line and updates the total distance odometer.
        """
        delta_ticks = (x1['right']['data'] - x0['right']['data'] + x1['left']['data'] - x0['left']['data']) / 2.0
        delta_ticks = delta_ticks * (pi/180)
        delta_time = ((x1['right']['time'] - x0['right']['time']) + (x1['left']['time'] - x0['left']['time'])) / 2.0
        self.velocity = self.radius * (delta_ticks/delta_time)
        self.last_distance = self.velocity * delta_time
        self.total_distance += self.last_distance
        if DEBUG:
            print("delta ticks: {} delta time: {} velocity: {} distance: {}".format(delta_ticks,
                                                                                    delta_time,
                                                                                    self.velocity,
                                                                                    self.last_distance))
