#!/usr/bin/env python3
from missions import Missions
import sys

x = list

def main(mission_num):

    missions = Missions()
    if mission_num == '1':
        missions.task_one()
    if mission_num == '2':
        missions.task_two()
    if mission_num == '3':
        missions.task_three(185, noisey=False)
    if mission_num == '4':
        missions.task_three(185, noisey=True)
    missions.exit_gracefully()


if __name__ == "__main__":
    main(sys.argv[1])
