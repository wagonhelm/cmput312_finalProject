from math import ceil, floor
from child_classes import StateList
import pickle
from sensor_distributions import mean_and_var

class BayesFilter:
    """Bayes filter used for discrete state estimation"""

    def __init__(self, color, noisey):
        # Environment variables
        self.track_length = 218.5  # cm
        self.discrete_size = ceil(self.track_length / 1.905) # 1.905 is our best estimate of a single 0.25 timestep
        self.discrete_step_size = self.track_length / self.discrete_size # Best true timestep size is 1.9
        gps_offset = 1.7  # Since color sensor is slightly in front of axles
        self.gps_points = [[30.6-gps_offset, 32.9-gps_offset],
                           [112.9-gps_offset, 115.2-gps_offset],
                           [167.3-gps_offset, 169.6-gps_offset]]

        if noisey is True:
            self.gps_error = 0.3
        else:
            self.gps_error = 0.01
        if self.gps_error == 0:
            self.gps_error_scale = 1e10
        else:
            self.gps_error_scale = (1-self.gps_error)/self.gps_error

        # Build Bayes map
        self.gps_map = StateList(size=self.discrete_size)
        self.time_step_gps_points = self.gps_points.copy()
        for i in self.time_step_gps_points:
            self.gps_map[floor(i[0] / self.discrete_step_size)] = 1
            self.gps_map[floor(i[1] / self.discrete_step_size)] = 1

        # Probabilities of states that can / cannot receive gps readings
        gps_prob = 1 / len(self.gps_points) / 2
        non_gps_prob = 1 / (self.discrete_size - len(self.gps_points) * 2)

        # Assign the equal probabilities of being in a gps / no gps state
        self.beliefs = {'gps': self.gps_map * gps_prob,
                        'no_gps': StateList(self.discrete_size, non_gps_prob) - (self.gps_map*non_gps_prob)}

        # Likelihoods are the non 1-sum values for the likilhood of receving a said sensor reading
        self.likelihoods = {'gps': StateList(self.discrete_size, 1/self.discrete_size),
                            'no_gps': StateList(self.discrete_size, 1/self.discrete_size)}
        for i in range(len(self.gps_map)):
            if self.gps_map[i] == 1:
                self.likelihoods['gps'][i] = self.gps_error_scale * self.likelihoods['gps'][i]
            else:
                self.likelihoods['no_gps'][i] = self.gps_error_scale * self.likelihoods['no_gps'][i]

        # Bayes filter variables
        self.kernel = [0.01, 0.95, 0.04]

        # Setup initial belief & Likelihood
        self.likelihood = None
        if color == "Red":
            self.belief = self.beliefs['gps']
        else:
            self.belief = self.beliefs['no_gps']

    def get_discrete_state(self, distance):
        """Given a current distance get a discrete state"""
        return floor(distance / self.discrete_step_size)

    def correct(self, color):
        """Dot products likelihood and belief to make belief match likelihood of sensor reading"""
        if color == "Red":
            self.likelihood = self.likelihoods['gps']
            self.belief = (self.likelihood * self.belief).normalize()
        else:
            self.likelihood = self.likelihoods['no_gps']
            self.belief = (self.likelihood * self.belief).normalize()

    def predict(self, delta_state):
        """Shifts the current belief using delta_state and distributing using the kernel"""
        self.belief = self.belief.apply_kernel(steps=delta_state, kernel=self.kernel)


class KalmanFilter:
    """Kalman filter used to estimate distance traveled"""

    def __init__(self, noisey=False):

        # Sensor error
        if noisey:
            self.odometry_var = mean_and_var['odom']['noisey_var']
        else:
            self.odometry_var = mean_and_var['odom']['var']
        self.gps = mean_and_var['gps']
        self.ultrasonic_var = mean_and_var['ultrasonic']['var']

        # State estimation
        self.mean = 0
        self.variance = 0
        self.counter = 1

        # Log data
        self.LOG_DATA = True
        self.max_data = 120
        self.data_log = {}
        # Create a bunch of entries in advance to hopefully speed things up
        for i in range(self.max_data):
            self.data_log[i] = {"current_mean": 0,
                                "current_var": 0,
                                "predict_mean": 0,
                                "predict_var": 0,
                                "lik_gps_mean": 0,
                                "lik_gps_var": 0,
                                "lik_us_mean": 0,
                                "lik_us_var": 0}

        # Which sensors to use
        self.color = True
        self.ultrasonic = True

    def predict(self, distance):
        """Increment current mean by distance traveled and add odometry variance to current variance"""
        self.mean += distance
        self.variance += self.odometry_var

        if self.LOG_DATA:
            self.data_log[self.counter]['predict_mean'] = self.mean
            self.data_log[self.counter]['predict_var'] = self.variance

    def update(self, current_data):
        # Apply gps likilihood
        if self.color:
            if current_data['color']['gps'] > 0:
                residual = current_data['color']['gps'] - self.mean
                kalman_gain = self.variance / (self.variance + self.gps[current_data['color']['count']].get('var'))
                self.mean = self.mean + kalman_gain*residual
                self.variance = (1-kalman_gain) * self.variance
                if self.LOG_DATA:
                    self.data_log[self.counter]['lik_gps_mean'] = self.mean
                    self.data_log[self.counter]['lik_gps_var'] = self.variance

        # Apply sonar
        if self.ultrasonic:
            residual = current_data['ultrasonic']['data'] - self.mean
            kalman_gain = self.variance / (self.variance + self.ultrasonic_var)
            self.mean = self.mean + kalman_gain * residual
            self.variance = (1 - kalman_gain) * self.variance
            if self.LOG_DATA:
                self.data_log[self.counter]['lik_us_mean'] = self.mean
                self.data_log[self.counter]['lik_us_var'] = self.variance

        self.data_log[self.counter]['current_mean'] = self.mean
        self.data_log[self.counter]['current_var'] = self.variance

        self.counter += 1

    def write_data_log(self):
        for i in range(self.counter, self.max_data):
            del self.data_log[i]
        """Saves data log to pickle file"""
        filename = 'data_log_kalman.pickle'
        with open(filename, 'wb') as fp:
            pickle.dump(self.data_log, fp)
        print("Pickled {}".format(filename))
