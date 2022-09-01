from waypoint_follower.waypoint_follow import PurePursuitPlanner
from expert_base import ExpertBase

class ExpertWaypointFollower(ExpertBase):
    def __init__(self, conf, wheelbase):
        planner = PurePursuitPlanner(conf, wheelbase)
    
    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        return self.planner.plan(pose_x, pose_y, pose_theta, lookahead_distance, vgain)

    