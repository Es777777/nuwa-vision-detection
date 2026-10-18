from enum import Enum

class TrackState(Enum):
    SEARCHING = 'SEARCHING'
    TENTATIVE = 'TENTATIVE'
    TRACKING = 'TRACKING'
    COASTING = 'COASTING'
    LOST = 'LOST'

class TrackFilter:
    """Small temporal gate used after RGB-D geometry, with no truth input."""
    def __init__(self, acquire=3, drop=5):
        self.state = TrackState.SEARCHING; self.hits = 0; self.misses = 0
        self.acquire = acquire; self.drop = drop
    def update(self, valid):
        if valid:
            self.hits += 1; self.misses = 0
            self.state = TrackState.TRACKING if self.hits >= self.acquire else TrackState.TENTATIVE
        else:
            self.misses += 1
            if self.misses >= self.drop: self.state = TrackState.LOST; self.hits = 0
            elif self.state in (TrackState.TRACKING, TrackState.TENTATIVE): self.state = TrackState.COASTING
        return self.state


class FieldMotionGate:
    """Reject impossible frame-to-frame jumps in a stable field frame."""
    def __init__(self, max_speed_mps=5.0, slack_m=0.10):
        self.max_speed_mps = float(max_speed_mps)
        self.slack_m = float(slack_m)
        self.last_position = None
        self.last_stamp = None

    def update(self, position, stamp):
        if position is None:
            return False, 'no_measurement'
        p = tuple(float(v) for v in position)
        t = float(stamp)
        if self.last_position is not None and self.last_stamp is not None:
            dt = t - self.last_stamp
            if dt > 0:
                distance = sum((a-b)**2 for a,b in zip(p,self.last_position))**0.5
                limit = self.max_speed_mps * dt + self.slack_m
                if distance > limit:
                    return False, f'jump_{distance:.3f}m_over_{limit:.3f}m'
        self.last_position = p
        self.last_stamp = t
        return True, 'accepted'
