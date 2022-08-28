from waypoint_follower.waypoint_follow import PurePursuitPlanner

class ExpertWaypointFollower():
    def __init__(self, conf, wheelbase):
        planner = PurePursuitPlanner(conf, wheelbase)
    
    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        return self.planner.plan(pose_x, pose_y, pose_theta, lookahead_distance, vgain)

    