import numpy as np


class DanceRevolutionStructure:
    def __init__(self, with_head=False, with_feet=False):
        # for some reason I need to change node 8 (which in the map is the root) with node 9 to match their skeleton
        self.head = [17, 15,  0, 16, 18]
        self.trunk_joints = [0, 1, 9]
        self.arm_joints = [4, 3, 2, 1, 5, 6, 7]
        self.leg_joints = [11, 10, 9, 12, 13, 14]

        self.feet = [23, 22, 24, 21, 19, 20]
        self.body = [self.trunk_joints, self.arm_joints, self.leg_joints]

        if with_head:
            self.body.append(self.head)

        if with_feet:
            self.body.append(self.feet)
