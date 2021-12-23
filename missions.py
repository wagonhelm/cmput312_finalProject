from ev3dev2.motor import MoveTank, OUTPUT_C, OUTPUT_D
import time
import localization
from sensors import Sensors
from odometry import Odometry
from ev3dev2.sound import Sound
from ev3dev2.button import Button
from ev3dev2.sensor.lego import TouchSensor
from sensors import average_timestep, adjust_timesteps
import signal

DEBUG = False


class Missions:

    def __init__(self):
        self.speaker = Sound()
        self.tank = MoveTank(OUTPUT_C, OUTPUT_D)
        self.sensors = None
        self.odometry = Odometry()
        self.button = Button()
        self.touch = TouchSensor()
        self.frequency = 0.25  # Rate at which you gather data
        self.kalman_filter = localization.KalmanFilter()

        # Kill motors and threads on sigterm
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        """Kills motors on sigterm/sigint"""
        self.tank.off()
        self.sensors.end_threads()
        self.sensors.write_data_log()
        self.kalman_filter.write_data_log()
        exit(78)

    def task_one(self):
        """
        First task is to localize using a bayes filter, put the train at any point and start the mission, if the train
        reaches the end of the track, it will stop, put train at the beginning of the track and start over.
        """
        self.sensors = Sensors(self.tank,max_data=500, discrete=True)
        self.speaker.tone([(392, 350, 100)])
        self.button.wait_for_bump("right")
        self.tank.on(15, 15)
        current_state = self.sensors.get_sensor_data_copy()
        start_time = time.time()

        # Infinite loop until sigterm sent
        while True:
            # Store previous state
            previous_state = current_state

            # Wait one time-step
            """ This seems to work better than sleep, and allows me to detect button presses in between timesteps"""
            while time.time()-average_timestep(previous_state) < self.frequency:
                if self.touch.is_pressed:
                    # Kill motors so we can reset
                    self.tank.off()

                    # Essentially pause the counter until we start the track over again
                    delta_time = time.time()-average_timestep(previous_state)
                    if DEBUG:
                        print("Button pressed at {}, {} seconds after timestep".format(time.time(), delta_time))
                        current_state = self.sensors.get_sensor_data_copy()

                        # Caluclate velocity and distance
                        self.odometry.calculate_velocity_and_distance(previous_state, current_state, DEBUG)

                        if DEBUG:
                            print("sync", average_timestep(previous_state, calculate_sync=True))
                            print(previous_state, "\n", current_state)
                            print("total_time", time.time() - start_time, "distance", self.odometry.total_distance)

                    # Wait to resume
                    self.button.wait_for_bump("right")
                    previous_state = adjust_timesteps(previous_state, -delta_time)
                    self.tank.on(15, 15)

            # Get current state, and make prediction using time difference
            current_state = self.sensors.get_sensor_data_copy()

            # Calculate velocity and distance
            self.odometry.calculate_velocity_and_distance(previous_state, current_state, DEBUG)

            if DEBUG:
                print("sync", average_timestep(previous_state, calculate_sync=True))
                print(previous_state, "\n", current_state)
                print("total_time", time.time() - start_time, "distance", self.odometry.total_distance)

    def task_two(self, distance=None, noisey=False):
        """
        Second task to use Kalman filter to fuse sensors and get a better sensor reading
        """
        self.sensors = Sensors(self.tank)
        self.kalman_filter = localization.KalmanFilter(noisey=noisey)
        done = False
        self.speaker.tone([(392, 350, 100)])
        self.button.wait_for_bump("right")
        current_state = self.sensors.get_sensor_data_copy()
        self.tank.on(15, 15)

        # Infinite loop until sigterm sent
        while done is False:
            # Store previous state
            previous_state = current_state

            # Wait one time-step
            """ This seems to work better than sleep, and allows me to detect button presses in between timesteps"""
            while time.time()-average_timestep(previous_state) < self.frequency:
                if self.touch.is_pressed:
                    # Kill motors so we can reset
                    self.tank.off()
                    done = True

            # Get current state, and make prediction using time difference
            current_state = self.sensors.get_sensor_data_copy()

            # Calculate velocity and distance
            self.odometry.calculate_velocity_and_distance(previous_state, current_state, DEBUG)

            # Make prediction (aka do movement using odometry)
            self.kalman_filter.predict(self.odometry.last_distance)

            # Do update
            self.kalman_filter.update(current_state)
            if distance:
                if self.kalman_filter.mean >= distance:
                    self.tank.off()
                    done = True

        print("Odometry Distance {}".format(self.odometry.total_distance))
        print("Kalman Distance {}".format(self.kalman_filter.mean))

    def task_three(self, distance, noisey):
        self.task_two(distance=distance, noisey=noisey)
