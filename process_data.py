from odometry import Odometry
from localization import BayesFilter
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sensors import average_timestep
import scipy.stats
from sensor_distributions import mean_and_var
import math
import sys

sensor_file = 'data/data.pickle'
kalman_file = 'data/data_kalman.pickle'


def process_discrete_data(noisey=False, plot=True):
    """Processes data for task one using bayes filter with option to plot data"""
    odometry = Odometry()

    plot = plot
    with open(sensor_file, 'rb') as fp:
        data = pickle.load(fp)

    # Setup start state
    for i in range(len(data)-1):
        if i == 0:
            bayes = BayesFilter(data[0]['color']['data'], noisey=noisey)
            current_data = data[0]
            current_state = 0
            x = [i for i in range(bayes.discrete_size)]

        # Progress one timestep
        previous_state = current_state
        previous_data = current_data
        current_data = data[i+1]

        # Calculate new discrete state based on odometry
        odometry.calculate_velocity_and_distance(previous_data, current_data)
        current_state = bayes.get_discrete_state(odometry.total_distance)
        delta_state = current_state-previous_state

        # Update bayes filter predictions
        gps_reading = current_data['color']['data']
        # Predict step
        bayes.predict(delta_state)
        # Correct step
        bayes.correct(gps_reading)

        # Plot results
        if plot is True:
            fig, ax = plt.subplots(figsize=(14, 9))
            plt.ylim(0, 1)
            plt.xlim(-0.5, bayes.discrete_size-0.5)
            ax2 = ax.twinx()
            ax2.axvline(x=15, color='red', label='Mock GPS')
            ax2.axvline(x=16, color='red')
            ax2.axvline(x=58, color='red')
            ax2.axvline(x=59, color='red')
            ax2.axvline(x=87, color='red')
            ax2.axvline(x=88, color='red')
            ax.set_yticks(np.arange(0, 1.1, 0.1))
            ax2.set_yticks(np.arange(0, 1.1, 0.1))
            ax.set_xticks(np.arange(0, bayes.discrete_size, 5))
            ax.set_ylabel('Probability')
            ax.set_xlabel('Discrete State')
            ax.set_title("Sensor: {}      Delta State: {}".format(gps_reading,delta_state))
            ax.bar(x, bayes.belief, width=1)
            ax.legend(labels=['Estimated State'], loc='upper right', bbox_to_anchor=(1, 0.95))
            plt.legend()
            plt.savefig("images/discrete_animation/bayes_distribution_{}.png".format(i))
            plt.clf()
            plt.close(fig)

    largest_value = max(bayes.belief)
    largest_index = bayes.belief.index(largest_value)
    print('Most likely state {}, distance {} of {} states with {} probability'.format(largest_index+1,
                                                                             (largest_index + 1)*1.9,
                                                                             len(bayes.belief),
                                                                             largest_value))

def plot_kalman_distributions():
    """Plot that kalman data for each timestep as a gaussian distribution"""
    x = np.arange(0, 200, 0.01)
    offset = 20

    with open(kalman_file, 'rb') as fp:
        kalman = pickle.load(fp)
    with open(sensor_file, 'rb') as fp:
        data = pickle.load(fp)

    for i in range(1, len(kalman) - 1):
        fig, ax = plt.subplots(figsize=(14, 9))

        # Progress one timestep
        sensor_data = data[i]
        kalman_data = kalman[i]
        previous_k_data = kalman[i-1]

        # Plot previous state
        previous_mean = previous_k_data['current_mean']
        previous_var = previous_k_data['current_var']
        previous_state = scipy.stats.norm.pdf(x, previous_mean, math.sqrt(previous_var))
        plt.plot(x, previous_state, label='previous_state', color='blue')

        # Plot prediction
        prediction = scipy.stats.norm.pdf(x, kalman_data['predict_mean'], math.sqrt(kalman_data['predict_var']))
        plt.plot(x, prediction, label='prediction', color='blue', linewidth=3, linestyle=':')
        plt.title("Prediction")
        plt.xlabel("Distance")
        plt.ylabel("Probability")
        plt.legend()
        plt.xlim(max(previous_mean - offset, 0), previous_mean + 10)
        plt.savefig("./images/kalman_distributions_animation/frame_{}_a.png".format(i))
        plt.clf()

        # Plot GPS Likilihood
        got_gps_data = False
        if kalman_data['lik_gps_mean'] != 0:
            likihood_mean = sensor_data['color']['gps']
            likihood_std = mean_and_var['gps']['std']
            gps_data = scipy.stats.norm.pdf(x, likihood_mean, likihood_std)
            got_gps_data = True

        # Plot ultrasonic Likilihood
        if kalman_data['lik_us_mean'] != 0:
            likihood_mean = sensor_data['ultrasonic']['data']
            likihood_std = mean_and_var['ultrasonic']['std']
            us_data = scipy.stats.norm.pdf(x, likihood_mean, likihood_std)

            # Plot prediction and all data
            plt.plot(x, previous_state, label='previous_state', color='blue')
            plt.plot(x, us_data, label='ultrasonic', color='green', linestyle=':')
            plt.plot(x, prediction, label='prediction', color='blue', linewidth=3, linestyle=':')
            if got_gps_data:
                plt.plot(x, gps_data, label='gps', color='red', linestyle=':')
            plt.title("Get Sensor Data")
            plt.xlabel("Distance")
            plt.ylabel("Probability")
            plt.legend()
            plt.xlim(max(previous_mean - offset, 0), previous_mean + 10)
            plt.savefig("./images/kalman_distributions_animation/frame_{}_b.png".format(i))
            plt.clf()


            # Update
            update_mean = kalman_data['lik_us_mean']
            update_std = math.sqrt(kalman_data['lik_us_var'])

            update = scipy.stats.norm.pdf(x, update_mean, update_std)
            plt.plot(x, previous_state, label='previous_state', color='blue')
            plt.plot(x, us_data, label='ultrasonic', color='green', linestyle=':')
            plt.plot(x, prediction, label='prediction', color='blue', linewidth=3, linestyle=':')
            if got_gps_data:
                plt.plot(x, gps_data, label='gps', color='red', linestyle=':')
            plt.plot(x, update, label='update', color='orange')
            plt.title("Do Update")
            plt.xlabel("Distance")
            plt.ylabel("Probability")
            plt.legend()
            plt.xlim(max(previous_mean - offset, 0), previous_mean + 10)
            plt.savefig("./images/kalman_distributions_animation/frame_{}_c.png".format(i))
            plt.clf()

        # Just plot current state and update
        plt.plot(x, previous_state, label='previous_state', color='blue')
        plt.plot(x, update, label='update', color='orange')
        plt.title("Update = Next State")
        plt.xlabel("Distance")
        plt.ylabel("Probability")
        plt.legend()
        plt.xlim(max(previous_mean - offset, 0), previous_mean + 10)
        plt.savefig("./images/kalman_distributions_animation/frame_{}_d.png".format(i))
        plt.clf()
        plt.close()



def plot_sensor_data(kalman_filter, animate):
    """Plot distance values for sensors and kalman filter, with open to create a graph for each timestep"""
    odometry = Odometry()

    with open(sensor_file, 'rb') as fp:
        data = pickle.load(fp)
    if kalman_filter is True:
        with open(kalman_file, 'rb') as fp:
            kalman = pickle.load(fp)

    start_time = average_timestep(data[0])
    total_time = average_timestep(data[len(data)-1]) - start_time

    # Plot
    fig, ax = plt.subplots()
    odo_x_points = [(data[i]['left']['time']+data[i]['right']['time'])/2-start_time for i in range(len(data))]
    gps_x_points = [data[i]['color']['time']-start_time for i in range(len(data))]
    us_x_points = [data[i]['ultrasonic']['time']-start_time for i in range(len(data))]
    kal_x_point = [average_timestep(data[i])-start_time for i in range(len(data))]

    # Data plots
    odometry_data = []
    ultrasonic_data = []
    gps_data = []
    kalman_data = []

    for i in range(len(data)-1):
        if i == 0:
            current_data = data[0]
            odometry_data.append(odometry.total_distance)
            ultrasonic_data.append(current_data['ultrasonic']['data'])
            gps_data.append(current_data['color']['gps'])
            kalman_data.append(kalman[i]['current_mean'])

        # Progress one timestep
        previous_data = current_data
        current_data = data[i + 1]

        # Plot Kalman
        kalman_data.append(kalman[i+1]['current_mean'])

        # Calculate distance based on odometry
        odometry.calculate_velocity_and_distance(previous_data, current_data)
        odometry_data.append(odometry.total_distance)

        # Calculate distance based on ultrasonic
        ultrasonic_data.append(current_data['ultrasonic']['data'])

        # Calculate distance based on gps
        gps_data.append(current_data['color']['gps'])

    if animate is False:
        # Plot points
        ax.grid()
        ax.scatter(odo_x_points, odometry_data, label="Odometry", color='Blue', s=8)
        ax.scatter(us_x_points, ultrasonic_data, label="Ultrasonic", color='Orange', s=8)
        ax.scatter(gps_x_points, gps_data, label="Mock GPS", color='Red', s=20)
        ax.plot(kal_x_point, kalman_data, label="Kalman Filter", linewidth=2, color='Green')
        ax.scatter(kal_x_point, kalman_data, s=8, color='Green')
        plt.legend()

        # Axis labels
        ax.set_yticks(np.arange(0, 231, 10))
        ax.set_xticks(np.arange(0, round(total_time), 5))
        plt.ylim(0, 230)
        plt.xlim(0, math.ceil(total_time))
        ax.set_ylabel('Distance (cm)')
        ax.set_xlabel('Time (s)')
        plt.savefig("images/all_sensor_data.png")
        plt.clf()
        fig.clear(True)
    else:
        max_index = len(kalman_data)-1
        for i in range(len(kalman_data)):
            if i - 5 > 0:
                index = i - 5
            else:
                index = 0
            fig, ax = plt.subplots(figsize=(14, 9))
            m_i = min(max_index, i+1)
            plt.xlim(kal_x_point[index], kal_x_point[m_i])
            ax.set_ylabel('Distance (cm)')
            ax.set_xlabel('Time (s)')
            ax.set_title("All sensor data")
            max_d = 0
            min_d = 300
            for distance in odometry_data[index:m_i] + ultrasonic_data[index:m_i] + kalman_data[index:m_i]:
                if distance > max_d:
                    max_d = distance
                if distance < min_d:
                    min_d = distance
            plt.ylim(min_d-5, max_d+5)
            plt.plot(kal_x_point[index:m_i], kalman_data[index:m_i], label="Kalman Filter", linewidth=2, color='Green')
            plt.scatter(kal_x_point[index:m_i], kalman_data[index:m_i], s=12, color='Green')
            plt.scatter(odo_x_points[index:m_i], odometry_data[index:m_i], label="Odometry", color='Blue', s=12)
            plt.scatter(us_x_points[index:m_i], ultrasonic_data[index:m_i], label="Ultrasonic", color='Orange', s=12)
            plt.scatter(gps_x_points[index:m_i], gps_data[index:m_i], label="Mock GPS", color='Red', s=12)
            plt.legend()
            plt.savefig("images/kalman_animation/kalman_{}.png".format(i))
            plt.clf()
            plt.close(fig)
    print('all done')

def timestep_error():
    with open(sensor_file, 'rb') as fp:
        data = pickle.load(fp)
    delta_times = []
    from data_collection import get_mean_and_std

    for i in range(len(data)-1):
        t = average_timestep(data[i])
        t2 = average_timestep(data[i+1])
        delta_times.append(t2-t)
    get_mean_and_std(delta_times)

def sync_error():
    with open(sensor_file, 'rb') as fp:
        data = pickle.load(fp)
    sync_times = []
    from data_collection import get_mean_and_std

    for i in data:
        max_time = 0
        min_time = 1e10
        for s in data[i]:
            s_time = data[i][s]['time']
            if s_time > max_time:
                max_time = s_time
            if s_time < min_time:
                min_time = s_time
        sync_times.append(max_time-min_time)
    get_mean_and_std(sync_times)




def main(arg):
    if arg == '1':
        process_discrete_data(plot=False)
    if arg == '1n':
        process_discrete_data(plot=False)

    if arg in ('2', '3', '4'):
        plot_sensor_data(kalman_filter=True, animate=False)

    if arg in ('2a', '3a', '4a'):
        plot_sensor_data(kalman_filter=True, animate=True)

    if arg in ('2k', '3k', '4k'):
        plot_kalman_distributions()


if __name__ == "__main__":
    main(sys.argv[1])
