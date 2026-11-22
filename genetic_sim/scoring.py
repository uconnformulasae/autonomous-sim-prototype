"""
@file scoring.py
@brief Everything about scoring cars lives here.

 - update_score(): the scoring formula; edit it and the constants below to change
   what the genetic run rewards
 - ScoreKeeper: tracks each car's score, distance, speed and crash time during
   a race, and builds the results dicts at the end
 - combine_scores(): turns a car's scores on several tracks into one ranking score

race() in testing_env.py makes a ScoreKeeper, calls step() after every physics step
and results() when the race is over.
"""

import numpy as np

from nn_controller import NUM_BEAMS, SPEED_MAX


# Each step earns up to SCORE_PER_STEP for distance covered (the full amount at
# SPEED_MAX), minus small penalties for bad speed choices.

SCORE_PER_STEP = 1.0
A_MAX = 9.51  # m/s^2, env's max acceleration; used to normalise deceleration
LIDAR_FOV = 4.7  # rad, env's default scan field of view
FRONT_HALF_ANGLE = np.radians(10)  # beams within this of straight ahead count as "ahead"
BRAKE_DECEL = 6.0  # m/s^2, braking assumed when working out a safe speed
STOP_MARGIN = 0.5  # m, gap to leave in front of the car when stopped
STRAIGHT_CLEARANCE = 10.0  # m, free space ahead that counts as a straight
FAST_PENALTY = 0.3  # per step, at a full SPEED_MAX over the safe speed
SLOW_PENALTY = 0.3  # per step, braking at A_MAX on a straight
CRASH_FACTOR = 0.5  # a crash keeps this fraction of the score

_front = int(FRONT_HALF_ANGLE / LIDAR_FOV * NUM_BEAMS)
FRONT_BEAMS = slice(NUM_BEAMS // 2 - _front, NUM_BEAMS // 2 + _front + 1)


def update_score(score, alive, crashed, scans, speed, prev_speed, step_dist, dt):
    """@brief Add this step's score to `score` in place. All arguments are arrays over cars.

    @param score      Running score per car (modified in place).
    @param alive      Cars still driving at the start of this step; only they score.
    @param crashed    Cars that crashed during this step (a subset of alive).
    @param scans      LiDAR ranges, shape (cars, NUM_BEAMS).
    @param speed      Speed now (m/s).
    @param prev_speed Speed one step ago (m/s).
    @param step_dist  Distance driven during this step (m).
    @param dt         Physics timestep (s).

    Free space straight ahead in the LiDAR tells a turn from a straight: a wall
    close ahead means a turn is coming. Too fast is faster than the car could
    stop in that space; on a straight, any braking is penalised. A crash cuts
    the score by CRASH_FACTOR.
    """
    ahead = scans[:, FRONT_BEAMS].min(axis=1)
    progress = SCORE_PER_STEP * step_dist / (SPEED_MAX * dt)

    safe_speed = np.sqrt(2 * BRAKE_DECEL * np.maximum(ahead - STOP_MARGIN, 0.0))
    too_fast = np.clip((speed - safe_speed) / SPEED_MAX, 0.0, 1.0)

    decel = np.clip((prev_speed - speed) / (A_MAX * dt), 0.0, 1.0)
    slowing_on_straight = np.where(ahead > STRAIGHT_CLEARANCE, decel, 0.0)

    step = progress - FAST_PENALTY * too_fast - SLOW_PENALTY * slowing_on_straight
    score[alive] += step[alive]
    score[crashed] *= CRASH_FACTOR


SCORE_FLOOR = 1e-3  # lowest per-track score that counts when multiplying across tracks


def combine_scores(per_track, tracks):
    """@brief Turn each car's per-track race scores into one score per car.

    @param per_track {track: [result dict per car]} as returned by race_tracks().
    @param tracks    The tracks to combine, in order.
    @return tuple (scores, geom_mean): scores is (len(tracks), n_cars), the raw
            score of each car on each track; geom_mean is (n_cars,), the number
            that ranks the cars.

    The scores are multiplied: geom_mean is the geometric mean, (s1 * s2 * ...)^(1/k),
    so a car has to do well on every track, and a bad track drags it down far more
    than it would in a plain average. Taking the k-th root keeps it on the same scale
    as a single track's score. Scores below SCORE_FLOOR (including negative ones,
    which would flip the sign of a product) count as SCORE_FLOOR, so a car that fails
    one track is nearly zeroed but can still be ranked by its other tracks.
    Edit this to change how tracks are combined.
    """
    scores = np.array([[r['score'] for r in per_track[t]] for t in tracks])
    geom_mean = np.exp(np.log(np.maximum(scores, SCORE_FLOOR)).mean(axis=0))
    return scores, geom_mean


class ScoreKeeper:
    """@brief Tracks every car's score and race statistics, and builds the final results.

    race() makes one per race, calls step() after every env.step(), and
    results() at the end. The arrays (alive, score, distance) are updated in
    place, so the render view can share them.
    """

    def __init__(self, n, obs, dt):
        """@brief Set up tracking for a new race.

        @param n   Number of cars.
        @param obs Observation from env.reset() (start positions and speeds).
        @param dt  Physics timestep (s).
        """
        self.dt = dt
        self.alive = np.ones(n, dtype=bool)
        self.death_time = np.full(n, np.nan)
        self.distance = np.zeros(n)
        self.score = np.zeros(n)
        self.speeds = [[] for _ in range(n)]
        self._xy = np.column_stack([obs['poses_x'], obs['poses_y']])
        self._speed = np.abs(np.asarray(obs['linear_vels_x'], dtype=np.float64))

    def step(self, obs, sim_time):
        """@brief Update scores and stats from the observation after one env step.

        @param obs      Observation returned by env.step().
        @param sim_time Simulated time so far (s); recorded as the crash time.
        """
        new_xy = np.column_stack([obs['poses_x'], obs['poses_y']])
        step_dist = np.linalg.norm(new_xy - self._xy, axis=1)
        self.distance[self.alive] += step_dist[self.alive]
        self._xy = new_xy
        prev_speed = self._speed
        self._speed = np.abs(np.asarray(obs['linear_vels_x'], dtype=np.float64))
        scans = np.asarray(obs['scans'], dtype=np.float64)
        crashed = self.alive & np.asarray(obs['collisions'], dtype=bool)
        update_score(self.score, self.alive, crashed, scans, self._speed, prev_speed,
                     step_dist, self.dt)
        for i in np.flatnonzero(self.alive):
            self.speeds[i].append(self._speed[i])
            if crashed[i]:
                self.alive[i] = False
                self.death_time[i] = sim_time

    def results(self, obs, sim_time):
        """@brief Build the results once the race is over.

        @param obs      Final observation (for lap counts).
        @param sim_time Total simulated time (s).
        @return list One dict per car: score, crashed, death_time_s, sim_time_s,
                distance_m, median_speed_mps, laps.
        """
        results = []
        for i in range(len(self.score)):
            crashed = not np.isnan(self.death_time[i])
            results.append({
                'score': round(self.score[i], 3),
                'crashed': crashed,
                'death_time_s': round(self.death_time[i], 3) if crashed else '',
                'sim_time_s': round(self.death_time[i] if crashed else sim_time, 3),
                'distance_m': round(self.distance[i], 3),
                'median_speed_mps': round(float(np.median(self.speeds[i])), 3) if self.speeds[i] else 0.0,
                'laps': int(obs['lap_counts'][i]),
            })
        return results
