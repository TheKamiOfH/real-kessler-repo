#define imminent asteroids
#better turning logic using only max turns
from kesslergame import KesslerController
#from collisions import circle_line_collision_continuous
from typing import Dict, Tuple
from pprint import pprint
import math
import numpy as np
from scipy.optimize import fsolve
global actions
actions = {0:{"doing":False, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0, "tts": 0  }}

def circle_circle_collision_time_interval(
    ax: float, ay: float, vax: float, vay: float, ra: float,
    bx: float, by: float, vbx: float, vby: float, rb: float
) -> tuple[float, float]:
    """
    Returns (t_enter, t_exit) if the two circles will collide,
    or (nan, nan) if there's no collision in the future.
    Can return (-inf, inf) if the circles collide always and ever.
    """
    # Courtesy of Jie and Kessler Game
    # This linalg version is mathematically the same as setting up a quadratic and solving it, but is faster since it simplifies things
    nan = None
    separation = ra + rb

    dx = ax - bx
    dy = ay - by
    dvx = vax - vbx
    dvy = vay - vby

    dist_sq = dx * dx + dy * dy
    speed_sq = dvx * dvx + dvy * dvy
    dot = dx * dvx + dy * dvy
    sep_sq = separation * separation

    # Both stationary. Either overlapping forever or never
    if math.isclose(speed_sq, 0.0):
        if dist_sq <= sep_sq:
            return -math.inf, math.inf # Always overlapping
        else:
            return nan, nan # Never collide

    # Already outside and moving away (or tangent and moving apart)
    if dot >= 0.0 and dist_sq > sep_sq:
        return nan, nan

    # sin check: if angle too wide, paths never intersect within radius band
    cos_theta_sq = (dot * dot) / (dist_sq * speed_sq)
    sin_theta_sq = 1.0 - cos_theta_sq
    min_sin_sq = sep_sq / dist_sq

    if sin_theta_sq > min_sin_sq:
        return nan, nan  # Will miss each other

    # Compute collision time interval centered around closest approach
    root_term = math.sqrt((sep_sq - dist_sq * sin_theta_sq) / speed_sq)
    t_mid = -dot / speed_sq

    t_enter = t_mid - root_term
    t_exit  = t_mid + root_term

    return t_enter, t_exit

def do_drop_mine(game_state, ship_state, current_frame):
    count = 0
    i_count = 0
    d_count = 0
    d_score = math.inf
    #add a check based on number of asteroids in close proximity
    if ship_state["mines_remaining"] <= 0:
        return (False, 0, 0,0)
    for asteroid in game_state["asteroids"]:
        score = coll_evaluate_asteroid(ship_state, asteroid)
        if score is not None:
            score *=30
        d_score = math.inf
        for i in range(30):
            a_x = asteroid["position"][0] + asteroid["velocity"][0]*i*(1/30)
            a_y = asteroid["position"][1] + asteroid["velocity"][1]*i*(1/30)
            cache = distance(a_x,a_y,ship_state["position"][0],ship_state["position"][1])
            if cache < d_score:
                d_score = cache
        if score is not None and score < 30:
            count+=1
        if score is not None and score < 2 :
            i_count += 1
        if d_score is not None and d_score < 250:
            d_count+=1
    if count >= 5 or i_count >= 1 or d_count>10:
        return (True, count,i_count,d_count)
    return (False,0,0,0)
def canonize(asteroid,current_frame):
    delta_t = 1/30
    new_a_x = round((asteroid["position"][0] - asteroid["velocity"][0] * current_frame * delta_t)%1000,2)
    new_a_y = round((asteroid["position"][1] - asteroid["velocity"][1] * current_frame * delta_t)%800,2)
    return (new_a_x, new_a_y, asteroid["size"])
def rot_evaluate_asteroid(ship_state, asteroid):
    theta1 = math.atan2(asteroid["position"][1]-ship_state["position"][1], asteroid["position"][0]-ship_state["position"][0]) # account for bullet spawn here
    theta1 = math.degrees(theta1)%360
    angle_diff = (theta1 - ship_state["heading"])%360
    if angle_diff > 180:
        angle_diff -= 360
    frame_diff = math.ceil(abs(angle_diff) / 6)
    return frame_diff

def coll_evaluate_asteroid(ship_state, asteroid):
    tup = circle_circle_collision_time_interval(ship_state["position"][0], ship_state["position"][1],0,0,ship_state["radius"],asteroid["position"][0], asteroid["position"][1], asteroid["velocity"][0], asteroid["velocity"][1], asteroid["radius"])
    check = None
    if tup[0] is not None: 
        if tup[0] < 0:
            if tup[1] is None or tup[1] < 0:
                return None
            check = tup[1]
        else:
            check = tup[0]  
    if check is not None:
        if tup[0] == math.inf or tup[0] == -math.inf:
            return -math.inf
        return check
    return None

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def fuzzy_threat_score(ship_state, asteroid):
    """Return a fuzzy threat in [0,1].

    Uses two simple linear memberships:
    - `close`: 1 at distance 0, 0 at distance >= 300
    - `closing`: projection of asteroid velocity toward the ship; 0 at <=0, 1 at >=200
    Fuzzy rule: threat = min(close, closing) (a simple fuzzy AND).
    """
    sx, sy = ship_state["position"]
    ax, ay = asteroid["position"]
    avx, avy = asteroid["velocity"]

    d = distance(sx, sy, ax, ay)
    close = max(0.0, min(1.0, (300.0 - d) / 300.0))

    if d > 1e-9:
        ux = (sx - ax) / d
        uy = (sy - ay) / d
        closing_speed = avx * ux + avy * uy
    else:
        closing_speed = 0.0

    closing = max(0.0, min(1.0, closing_speed / 200.0)) if closing_speed > 0 else 0.0
    return min(close, closing)

def can_shoot(actions,frame):
    for i in range(frame-3, frame):
        if i in actions.keys() and actions[i]["shooting"]:
            return False
    return True

def _bullet_hit_time(bx, by, bvx, bvy, ax, ay, avx, avy, arad, life):
    """Time at which a bullet (point at its head) first strikes the asteroid,
    or None if it won't hit within the bullet's remaining lifetime `life`."""
    te, tx = circle_circle_collision_time_interval(bx, by, bvx, bvy, 0.0,
                                                   ax, ay, avx, avy, arad)
    if te is None:
        return None
    if te == -math.inf:
        return 0.0
    if tx is None or tx < -1e-9:
        return None
    if te > life:
        return None
    return max(0.0, te)

def doomed_asteroid_indices(asteroids, bullets, map_size, ship_state):
    """Return the set of asteroid indices that an in-flight bullet (ours OR the
    opponent's) is already on course to destroy. Targeting these wastes a shot.

    A bullet is consumed by the FIRST asteroid it hits, so each bullet marks at
    most one asteroid as doomed (the earliest along its path)."""
    W, H = map_size
    doomed = set()
    if not bullets or not asteroids:
        return doomed
    sx0, sy0 = ship_state["position"]
    for b in bullets:
        bx, by = b["position"]
        bvx, bvy = b["velocity"]
        # Remaining lifetime: time until the bullet head exits the map
        life = math.inf
        if bvx > 1e-9:
            life = min(life, (W - bx) / bvx)
        elif bvx < -1e-9:
            life = min(life, (0.0 - bx) / bvx)
        if bvy > 1e-9:
            life = min(life, (H - by) / bvy)
        elif bvy < -1e-9:
            life = min(life, (0.0 - by) / bvy)
        if life == math.inf:
            life = 2.0
        life = max(0.0, life)
        best_idx = None
        best_t = math.inf
        for i, a in enumerate(asteroids):
            ax, ay = a["position"]
            avx, avy = a["velocity"]
            t = _bullet_hit_time(bx, by, bvx, bvy, ax, ay, avx, avy, a["radius"], life)
            if t is not None and t < best_t:
                best_t = t
                best_idx = i
        # Only treat size-1 asteroids as "doomed": destroying them spawns no
        # children, so skipping a re-shot is pure savings with no survival risk.
        # Larger asteroids split into children that may still threaten us, so we
        # keep the option to re-engage them. Also never skip something that's on
        # a collision course with us -- if that en-route bullet misses, we still
        # want it as a target.
        if (best_idx is not None and asteroids[best_idx]["size"] == 1
                and coll_evaluate_asteroid(ship_state, asteroids[best_idx]) is None):
            ba = asteroids[best_idx]
            bax, bay = ba["position"]
            if (bax - sx0) * (bax - sx0) + (bay - sy0) * (bay - sy0) > 120.0 * 120.0:
                doomed.add(best_idx)
    return doomed

def aim_bot(s_x,s_y,a_x,a_y,a_vx,a_vy,s_h) -> None:
    "Update the actions dict wih the appropriate turning and shooting actions"
    delta_t = 1/30
    current_frame = -1
    for i in actions.keys():
        if actions[i]["doing"]:
            continue
        else:
            current_frame = i
    if current_frame == -1:
        current_frame = max(actions.keys()) + 1
    for f in  range(0,330):
        last_thing = False
        new_a_x = (a_x + a_vx * f * delta_t)%1000
        new_a_y = (a_y + a_vy * f * delta_t)%800
        theta1 = math.atan2(new_a_y-s_y, new_a_x-s_x) # account for bullet spawn here
        theta1 = math.degrees(theta1)%360
        angle_diff = (theta1 - s_h)%360
        if angle_diff > 180:
            angle_diff -= 360
        sign = 1 if angle_diff > 0 else -1
        #turning_frames = math.ceil(abs(angle_diff) / 6)
        shooting_frames = math.ceil(distance(s_x, s_y, new_a_x, new_a_y) / (800/30))
        #rate = max(-180, min(180, angle_diff*30))
        
        if f < shooting_frames:
            continue
        turning_frames = f - shooting_frames
        if turning_frames <= 0:
            turning_frames = 1
        if abs(angle_diff)/turning_frames > 6:
            continue
        max_turn_frames = int(abs(angle_diff)/6)
        
        #max_turn_frames = math.floor(abs(angle_diff) / 6)
        if 6*max_turn_frames < abs(angle_diff):
            last_thing = True
        if last_thing:
            turning_frames = max_turn_frames + 1
        else:
            turning_frames = max_turn_frames
        if not can_shoot(actions,current_frame + turning_frames + shooting_frames):
            #print("Triggtered no shoot")
            continue
        if turning_frames + shooting_frames <= f:
            #decide what to do for the frames before we start turning
            extra_frames = f - max_turn_frames - shooting_frames
            for i in range(extra_frames):
                if current_frame + i not in actions.keys():
                    actions[current_frame + i] = {"doing":True, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0,"tts":0}
                action = actions[current_frame + i]
                action["doing"] = True
                action["turning"] = False
                action["shooting"] = False
            #update actions dict
            for i in range(max_turn_frames):
                #actions[current_frame + i] = {"turning":True, "shooting":False, "dropping_mine":False, "turn": sign * 6,}
                if current_frame + extra_frames + i not in actions.keys():
                    actions[current_frame + extra_frames + i] = {"doing":True, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0,"tts":0}
                action= actions[current_frame + extra_frames + i]
                action["doing"] = True
                action["turning"] = True
                action["turn"] = sign *6*30
            if last_thing: #and actions[current_frame + extra_frames + max_turn_frames]["tts"] == 0 or actions[current_frame + extra_frames + max_turn_frames]["tts"] == 1:
                if current_frame + max_turn_frames + extra_frames + 1 not in actions.keys():
                    actions[current_frame + max_turn_frames + extra_frames + 0 ] = {"doing":True, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0,"tts":0}
            
                action= actions[current_frame + max_turn_frames + extra_frames + 0]
           
           
                #still some turning to do after max turn frames, so shoot while turning
                action["turning"] = True
                action["doing"] = True
                action["turn"] = sign * (abs(angle_diff) - 6*max_turn_frames)*30
                action["shooting"] = True
                action["tts"] = 3
                '''
                for i in range(current_frame + extra_frames + max_turn_frames + 1, current_frame + extra_frames + max_turn_frames + 3):
                    if i not in actions.keys():
                        actions[i] = {"doing":False, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0,"tts":0}
                    action = actions[i]
                    prev_action = actions[i-1]
                    action["tts"] = prev_action["tts"] - 1
                '''
                break
            elif (not last_thing): #and actions[current_frame + extra_frames + max_turn_frames]["tts"] == 0 or actions[current_frame + extra_frames + max_turn_frames]["tts"] == 1:
                # Shoot on the last turning frame. If the target is already aligned
                # (no turn frames), shoot on the current aligned frame instead of
                # indexing one before the start (which would be a negative key).
                shoot_idx = current_frame + max_turn_frames + extra_frames - 1
                if shoot_idx < current_frame:
                    shoot_idx = current_frame + max_turn_frames + extra_frames
                if shoot_idx not in actions:
                    actions[shoot_idx] = {"doing":True, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0,"tts":0}
                action = actions[shoot_idx]
                action["doing"] = True
                action["shooting"] = True
                '''
                for i in range(current_frame + extra_frames + max_turn_frames + 1, current_frame + extra_frames + max_turn_frames + 3):
                    if i not in actions.keys():
                        actions[i] = {"doing":False, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0,"tts":0}
                    action = actions[i]
                    prev_action = actions[i-1]
                    action["tts"] = prev_action["tts"] - 1
                '''
                break
            else:
                continue
        else:
            continue





def mine_blast_payoff(asteroids, sx, sy, lookahead_frames, map_size):
    """How many asteroids will sit inside a mine's blast (dropped now at the
    ship's position) when it detonates `lookahead_frames` from now.

    Matches the engine exactly: the mine sits at a fixed point (the stationary
    ship's position), asteroids wrap, and the blast uses direct (non-wrapped)
    distance with a (blast_radius + asteroid_radius) threshold."""
    W, H = map_size
    dt = 1.0 / 30.0
    BLAST = 150.0
    t = lookahead_frames * dt
    count = 0
    for a in asteroids:
        ax, ay = a["position"]
        avx, avy = a["velocity"]
        fx = (ax + avx * t) % W
        fy = (ay + avy * t) % H
        dx = fx - sx
        dy = fy - sy
        reach = BLAST + a["radius"]
        if dx * dx + dy * dy <= reach * reach:
            count += 1
    return count


def detect_closing_ring(asteroids, sx, sy, map_size,
                        horizon_frames=420, blast=150.0, frac=0.5, min_asts=8):
    """Detect a 'closing ring': a large fraction of asteroids that will all be
    inside a single mine blast of the ship at roughly the same future moment --
    i.e. a cluster converging onto the (stationary) ship. Returns
    (is_ring, peak_count, peak_frame)."""
    n = len(asteroids)
    if n < min_asts:
        return (False, 0, 0)
    W, H = map_size
    dt = 1.0 / 30.0
    snap = [(a["position"][0], a["position"][1],
             a["velocity"][0], a["velocity"][1], a["radius"]) for a in asteroids]
    peak = 0
    peak_f = 0
    for f in range(0, horizon_frames + 1, 3):   # 3-frame steps for speed
        t = f * dt
        c = 0
        for ax, ay, vx, vy, r in snap:
            fx = (ax + vx * t) % W
            fy = (ay + vy * t) % H
            dx = fx - sx
            dy = fy - sy
            reach = blast + r
            if dx * dx + dy * dy <= reach * reach:
                c += 1
        if c > peak:
            peak = c
            peak_f = f
    return (peak >= frac * n, peak, peak_f)


def find_opponent(game_state, my_id):
    """Return the other (live) ship's view, or None if we're alone."""
    for s in game_state["ships"]:
        if s["id"] != my_id:
            return s
    return None


def navigate_to(ship_state, tx, ty, map_size, stop=True):
    """Steer the ship toward (tx, ty). With stop=True, brake to rest on arrival;
    with stop=False, keep accelerating in (used to RAM a target).
    Returns (thrust, turn_rate, distance_to_target)."""
    x, y = ship_state["position"]
    W, H = map_size
    dx = tx - x
    dy = ty - y
    # Take the shortest path across the wrap-around edges
    if dx > W * 0.5:
        dx -= W
    elif dx < -W * 0.5:
        dx += W
    if dy > H * 0.5:
        dy -= H
    elif dy < -H * 0.5:
        dy += H
    dist = math.hypot(dx, dy)
    desired = math.degrees(math.atan2(dy, dx)) % 360.0
    err = (desired - ship_state["heading"] + 180.0) % 360.0 - 180.0
    turn = max(-180.0, min(180.0, err * 12.0))
    speed = ship_state["speed"]
    brake_dist = speed * speed / (2.0 * 450.0) + 3.0   # distance needed to stop (small margin = settle right on them)
    if abs(err) > 30.0:
        thrust = 0.0          # turn to face the target before burning
    elif (not stop) or dist > brake_dist:
        thrust = 480.0        # accelerate toward it (always, if ramming)
    else:
        thrust = -480.0       # brake so we come to rest on top of them
    return thrust, turn, dist


class TEController(KesslerController):
   # Drop a mine when at least this many asteroids will be caught in the blast
   # at detonation time. Tunable; swept empirically.
   MINE_K = 5
   # Only add a proactive early-mine when at least this many asteroids are on a
   # collision course with us within 3s -- i.e. we're being overwhelmed and would
   # die anyway. Keeps proactive mining out of scenarios we'd otherwise survive.
   MINE_THREAT = 3
   # --- DEATH BLOSSOM closing-ring choreography (exactly 2 mines) ---
   # Both mines are dropped on a peak-collapse, delayed this many extra frames so the
   # blast detonates a touch LATER -- by then the ring (shoved outward by our bullet
   # hits) has closed back into blast range and the mine actually connects instead of
   # blowing up in empty space.
   RING_MINE_DELAY = 20
   # Drop a mine only if the blast would catch at least this many asteroids (so we
   # fire onto a packed cluster, never a scattered/empty field).
   RING_MINE_MINCOUNT = 6
   # Mine #1 fires on the incoming big ring. Mine #2 (the finisher) is held until the
   # big asteroids are mostly gone -- at most this many size>=3 remain -- so it lands on
   # the collapsed remnant instead of prematurely blowing the ring apart. Lower = wait
   # longer before the finisher arms.
   RING_FINISH_BIG = 2
   # Last-instant finisher: how many frames BEFORE an incoming asteroid would hit us to
   # drop the mine. Because the 3s mine fuse equals the 3s respawn-invuln, the invuln
   # left AFTER the blast == this lead. So a bigger lead = more post-blast invuln for the
   # blasted asteroids to clear before we're vulnerable again = safer (at the cost of the
   # mine detonating a little later relative to the ring). Raise this if we still die
   # right after the blast.
   RING_FINISHER_LEAD = 5
   # --- RING STEALING ---
   # If the OPPONENT is trapped in a closing ring and we are NOT, fly onto them and
   # mine them. We may mine once we're within this many units of the opponent.
   STEAL_MINE_DIST = 80.0
   # Keep thrusting onto the opponent (even after a hit resets our speed) until we are
   # this close -- so we fully overlap/center on them rather than stalling short.
   STEAL_CENTER_DIST = 5.0
   # Patience: while we still have plenty of invulnerability we DON'T mine -- we use
   # those frames to stay alive and centered. Only drop a mine once the invuln has
   # almost run out (this many seconds left) or we're already vulnerable. Spends the
   # full invuln window each life.
   STEAL_MINE_INVULN = 0.15
   # --- AMMO-DEPLETED behavior (out of bullets) ---
   # When out of bullets but holding mines: drop a mine right here if its blast would
   # catch at least this many asteroids, otherwise drive to the densest cluster first.
   AMMO_MINE_MIN = 3
   def __init__(self):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        ...
        #self.dead_asteroids_dict = {} # Dictionary to keep track of dead asteroids
        self.d_list = [] #list of dead asteroids
        self.sequence = {}
        self.current_frame = 0
        self.closest_asteroid ={}
        self.targeted = False
        self.Icount = 0
        self.num_mines_to_drop = 0
        self.mine_dropped = False
        self.last_dropped_mine_frame = -100
        self.frame_to_drop = 0 # Track when the last mine was dropped
        self.dropped_mine_cuz_scared = -100  # Flag to indicate if a mine was dropped due to being scared
        # DEATH BLOSSOM state
        self.ring_mode = False
        self.ring_checked = False
        self.ring_mines_dropped = 0   # how many choreography mines dropped so far (cap 2)
        self.ring_pending = -1        # frame to drop the next (delayed) mine, or -1
        # RING STEALING state + explainability
        self.ring_steal = False
        self._steal_started = False
        self._said = set()            # keys of one-time explanation messages already printed
        # AMMO-DEPLETED state (cached densest-cluster target)
        self._dense_target = None
        self._dense_frame = -999
   def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
       if game_state["frame"]==0:
            self.d_list = [] #list of dead asteroids
            self.sequence = {}
            self.current_frame = 0
            self.closest_asteroid ={}
            self.targeted = False
            self.Icount = 0
            self.num_mines_to_drop = 0
            self.mine_dropped = False
            self.last_dropped_mine_frame = -100
            self.frame_to_drop = 0 # Track when the last mine was dropped
            self.dropped_mine_cuz_scared = -100  # Flag to indicate if a mine was dropped due to being scared
            self.ring_mode = False
            self.ring_checked = False
            self.ring_mines_dropped = 0
            self.ring_pending = -1
            self.ring_steal = False
            self._steal_started = False
            self._said = set()
            self._dense_target = None
            self._dense_frame = -999
            global actions
            actions = {0:{"doing":False, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0, "tts": 0  }}
       invinc = ship_state["respawn_time_left"]
       # --- Strategy detection (once, at the start) ---------------------------
       if not self.ring_checked:
           self.ring_checked = True
           asts0 = game_state["asteroids"]
           msize0 = game_state["map_size"]
           is_ring, _peak, _pf = detect_closing_ring(
               asts0, ship_state["position"][0], ship_state["position"][1], msize0)
           self.ring_mode = is_ring
           if is_ring:
               self._say("detect", "DEATH BLOSSOM: a closing ring is converging on us -> ring mode engaged.")
           else:
               # We're free -- is the OPPONENT stuck in a closing ring we can steal?
               opp = find_opponent(game_state, ship_state["id"])
               if opp is not None:
                   opp_ring, _p2, _f2 = detect_closing_ring(
                       asts0, opp["position"][0], opp["position"][1], msize0)
                   if opp_ring:
                       self.ring_steal = True
                       self._say("detect", "RING STEAL: the opponent is trapped in a closing ring and we are free "
                                           "-> diving into their ring to mine them and deny their harvest.")
               if not self.ring_steal:
                   self._say("detect", "No closing ring in play -> standard shooting behavior.")
       # --- RING STEAL: fly onto the trapped opponent and mine them -----------
       if self.ring_steal:
           thrust, turn, fire, drop_mine = self._ring_steal_action(ship_state, game_state)
           self.current_frame += 1
           return thrust, turn, fire, drop_mine
       # --- OUT OF BULLETS: mine the densest spot, or ram asteroids -----------
       if ship_state["bullets_remaining"] == 0 and not self.ring_mode and game_state["asteroids"]:
           thrust, turn, fire, drop_mine = self._ammo_depleted_action(ship_state, game_state)
           self.current_frame += 1
           return thrust, turn, fire, drop_mine
       doing = True if self.current_frame in actions and actions[self.current_frame]["doing"] else False
       if len(self.d_list) > 2 or len(self.d_list) >= len(game_state["asteroids"]) - 1:
           self.d_list = self.d_list[1:]
       if not doing:

           target = None
           score = 100
           asteroid_list = game_state["asteroids"]
           # Skip asteroids that an in-flight bullet (ours or the opponent's) will already destroy
           doomed = doomed_asteroid_indices(
               asteroid_list, game_state["bullets"], game_state["map_size"], ship_state)
           for a_idx, asteroid in enumerate(asteroid_list):
                # Defensive pass: target asteroids on a collision course with us.
                # We do NOT skip "doomed" ones here -- if something is coming at us,
                # a redundant bullet is cheap insurance against a missed/splitting kill.
                key = canonize(asteroid,self.current_frame)
                if key not in self.d_list:
                     cache = coll_evaluate_asteroid(ship_state, asteroid)
                     if cache is not None:
                         if cache*30<score*30:
                            score = cache
                            target = asteroid
           if target is None:
                for a_idx, asteroid in enumerate(asteroid_list):
                    if a_idx in doomed:
                        continue
                    key = canonize(asteroid,self.current_frame)
                    if key not in self.d_list:
                        cache = rot_evaluate_asteroid(ship_state, asteroid)
                        if cache is not None and cache < score:
                            score = cache
                            target = asteroid
           if target is not None:
                       print("Target Selected")
                       canon_pos = canonize(target,self.current_frame)
                       self.d_list.append(canon_pos)
                       aim_bot(ship_state["position"][0], ship_state["position"][1], target["position"][0], target["position"][1], target["velocity"][0], target["velocity"][1], ship_state["heading"])
                       if target["size"]>1:#if big boi wait a frame
                           i = len(actions.keys())
                           i+=1
                           actions[i] = {"doing":True, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0, "tts": 0  }
           #print(actions)
       fire = actions[self.current_frame]["shooting"] if self.current_frame in actions else False
       if not fire: #spray and pray check
           if ship_state["bullets_remaining"] > 50 or ship_state["bullets_remaining"]==-1:
               if self.current_frame+3 not in actions.keys() or (not actions[self.current_frame+3]["shooting"]):
                   fire = True
       turn = actions[self.current_frame]["turn"] if self.current_frame in actions else 0
       thrust = 0
       drop_mine = False
       # Decide the ring-mode invuln RIDE-OUT first, so the choreography below never
       # "spends" a mine that the ride-out would then cancel (that was the bug that
       # made the finisher never deploy). Both firing and dropping a mine cancel invuln,
       # so while riding out we do NEITHER and let the invulnerability soak the blast.
       riding_out = False
       if invinc > 0 and self.ring_mode:
           danger = False
           for asteroid in game_state["asteroids"]:
               ttc = coll_evaluate_asteroid(ship_state, asteroid)
               if ttc is not None and ttc <= invinc:   # an asteroid hits within invuln
                   danger = True
                   break
           if not danger:
               danger = self._mine_threat_within(ship_state, game_state, invinc)
           riding_out = danger

       if self.ring_mode and ship_state["bullets_remaining"] == -1:
           # Exactly two mines:
           #   * Mine #1 hits the incoming big ring (slows it; armed immediately, fired
           #     on the collapse crest + a short delay so it connects).
           #   * Mine #2 (the finisher) is held until the big asteroids are mostly gone,
           #     then dropped onto the collapsed remnant.
           # Plus a last-instant DEFENSIVE drop: if we're about to be hit and still hold
           # a mine, drop it NOW rather than die with it unused.
           # Skipped while riding out invuln (a drop would cancel the invuln).
           if not riding_out:
               sx, sy = ship_state["position"]
               asts = game_state["asteroids"]
               msize = game_state["map_size"]
               n_big = sum(1 for a in asts if a["size"] >= 3)   # genuinely-big asteroids
               can = ship_state["can_deploy_mine"] and ship_state["mines_remaining"] != 0
               big_mostly_gone = n_big <= self.RING_FINISH_BIG
               # Wait for the previous mine to detonate (~90f) so they don't stack.
               prev_detonated = (self.last_dropped_mine_frame < 0
                                 or self.current_frame - self.last_dropped_mine_frame >= 90)
               if can and self.ring_mines_dropped < 2:
                   # Mine #1 is always armed; the finisher arms once big asteroids gone.
                   armed = (self.ring_mines_dropped == 0) or big_mostly_gone
                   if prev_detonated and armed:
                       payoff_now = mine_blast_payoff(asts, sx, sy, 90, msize)
                       if payoff_now >= self.RING_MINE_MINCOUNT:
                           if self.ring_mines_dropped == 0:
                               # Mine #1: wait for the incoming collapse to crest.
                               ready = payoff_now >= mine_blast_payoff(asts, sx, sy, 93, msize)
                           else:
                               # Finisher: remnant is already slowed/near -- just go.
                               ready = True
                           if ready and self.ring_pending < 0:
                               self.ring_pending = self.current_frame + self.RING_MINE_DELAY
                       if self.ring_pending >= 0 and self.current_frame >= self.ring_pending:
                           drop_mine = True
                           if self.ring_mines_dropped == 0:
                               self._say_mine("DEATH BLOSSOM mine #1 -> detonating as the ring collapses onto us, to slow/clear it.")
                           else:
                               self._say_mine("DEATH BLOSSOM finisher -> big asteroids gone, dropping on the collapsed remnant.")
                           self.ring_pending = -1
                           self.ring_mines_dropped += 1
                           self.last_dropped_mine_frame = self.current_frame
                   # Defensive last-instant finisher: about to be hit and still holding a
                   # mine -> drop it RING_FINISHER_LEAD frames before the hit, so we keep
                   # that many frames of invuln after the blast to ride out the fallout.
                   if not drop_mine and prev_detonated:
                       for a in asts:
                           ttc = coll_evaluate_asteroid(ship_state, a)
                           if ttc is not None and ttc * 30.0 <= self.RING_FINISHER_LEAD:
                               drop_mine = True
                               self._say_mine(f"DEATH BLOSSOM finisher (defensive) -> about to be hit; dropping {int(self.RING_FINISHER_LEAD)}f early, then riding out invuln.")
                               self.ring_pending = -1
                               self.ring_mines_dropped += 1
                               self.last_dropped_mine_frame = self.current_frame
                               break
       else:
           # (1) Defensive last-instant mining.
           to_drop = do_drop_mine(game_state, ship_state, self.current_frame)
           if to_drop[0]:
               if to_drop[2] >= 1 and self.current_frame - self.dropped_mine_cuz_scared >= 100:
                   self.last_dropped_mine_frame = self.current_frame
                   self.dropped_mine_cuz_scared = self.current_frame
                   drop_mine = True
           # (2) Scarce-ammo proactive peak mining.
           if (not drop_mine and ship_state["mines_remaining"] != 0
                   and ship_state["can_deploy_mine"]
                   and ship_state["bullets_remaining"] != -1
                   and ship_state["bullets_remaining"] < len(game_state["asteroids"])
                   and self.current_frame - self.last_dropped_mine_frame > 30):
               sx, sy = ship_state["position"]
               asts = game_state["asteroids"]
               msize = game_state["map_size"]
               threat = 0
               fuzzy_threat = 0.0
               for a in asts:
                   ttc = coll_evaluate_asteroid(ship_state, a)
                   if ttc is not None and ttc <= 3.0:
                       threat += 1
                   # accumulate trivial fuzzy score per asteroid
                   fuzzy_threat += fuzzy_threat_score(ship_state, a)
               # Trigger when either the integer threat count or the summed fuzzy
               # threat crosses the same numeric threshold
               if threat >= self.MINE_THREAT or fuzzy_threat >= float(self.MINE_THREAT):
                   payoff_now = mine_blast_payoff(asts, sx, sy, 90, msize)
                   payoff_next = mine_blast_payoff(asts, sx, sy, 91, msize)
                   if payoff_now >= self.MINE_K and payoff_now >= payoff_next:
                       drop_mine = True
                       self.last_dropped_mine_frame = self.current_frame
       # End-of-game dump: spend a leftover mine before the scenario ends
       if (not riding_out
               and game_state["time_limit"] != math.inf
               and game_state["time_limit"]*30 - self.current_frame <= 91
               and len(game_state["asteroids"])>5):
           if not drop_mine and ship_state["can_deploy_mine"] and ship_state["mines_remaining"] != 0:
               self._say_mine("End of scenario -> dumping leftover mine(s) for free hits, no reason to save them.")
           drop_mine = True
       if invinc > 0 and not self.ring_mode:
            # Non-ring: preserve invuln by not firing into a near threat.
            count = 0
            for asteroid in game_state["asteroids"]:
                ttc = coll_evaluate_asteroid(ship_state,asteroid)
                if ttc is not None:
                    if ttc*30<35:
                        count +=1
            # Also ride out invuln if our own mine is still in the air over us 
            mine_threat = self._mine_threat_within(ship_state, game_state, invinc)
            if mine_threat:
                fire = False
                drop_mine = False
                self._say("rideout_mine", "Our mine is still ticking over us -> holding fire to ride out the full invuln until it blows.")
            if count>0:
                fire = False
                drop_mine = False
                if len(self.d_list)>0:
                    self.d_list.pop()
                if self.last_dropped_mine_frame == self.current_frame:
                    self.last_dropped_mine_frame = -100
                if self.dropped_mine_cuz_scared == self.current_frame:
                    self.dropped_mine_cuz_scared = -100
       elif riding_out:
            # Ring mode ride-out hold fire an mine so the invuln soaks the blast/impact.
            self._say("rideout", "Hit while a mine/asteroid is closing -> holding fire to ride out invulnerability through the blast.")
            fire = False
            drop_mine = False

       # Last-life last-ditch mine: if we're on our final life, vulnerable, about to be
       # hit, and still holding a mine -> drop it right before we die
       if (not drop_mine and invinc <= 0.0
               and ship_state["lives_remaining"] <= 1
               and ship_state["can_deploy_mine"] and ship_state["mines_remaining"] != 0):
           for a in game_state["asteroids"]:
               ttc = coll_evaluate_asteroid(ship_state, a)
               if ttc is not None and ttc * 30.0 <= 5.0:   # about to be hit (~5 frames)
                   drop_mine = True
                   self._say_mine("Last life and about to be hit -> dropping our final mine before we die.")
                   break

       #drop_mine = actions[self.current_frame]["dropping_mine"] if self.current_frame in actions else False
       self.current_frame += 1
       return thrust, turn, fire, drop_mine

   def _say(self, key, msg):
       """Print a one-time explanation (deduped by key)."""
       if key not in self._said:
           self._said.add(key)
           print(f"[T.E. f{self.current_frame}] {msg}")

   def _say_mine(self, msg):
       """Print a per-drop mine explanation."""
       print(f"[T.E. f{self.current_frame}] MINE: {msg}")

   def _mine_threat_within(self, ship_state, game_state, horizon):
       """True if any mine in the air will detonate within `horizon` seconds and its
       blast covers the ship -- i.e. a mine we should ride out (or not re-expose to)."""
       sxp, syp = ship_state["position"]
       reach = 150.0 + ship_state["radius"]
       for m in game_state["mines"]:
           if m["remaining_time"] <= horizon:
               mdx = m["position"][0] - sxp
               mdy = m["position"][1] - syp
               if mdx * mdx + mdy * mdy <= reach * reach:
                   return True
       return False

   def _densest_point(self, asteroids, map_size):
       """Position of the asteroid with the most neighbors inside a mine blast radius
       (a good cluster center to mine). O(n^2), so cache and recompute periodically."""
       if self._dense_target is not None and self.current_frame - self._dense_frame < 15:
           return self._dense_target
       self._dense_frame = self.current_frame
       pts = [(a["position"][0], a["position"][1]) for a in asteroids]
       n = len(pts)
       r2 = 150.0 * 150.0
       best_i, best_c = -1, -1
       for i in range(n):
           xi, yi = pts[i]
           c = 0
           for xj, yj in pts:
               dx = xj - xi
               dy = yj - yi
               if dx * dx + dy * dy <= r2:
                   c += 1
           if c > best_c:
               best_c, best_i = c, i
       self._dense_target = pts[best_i] if best_i >= 0 else None
       return self._dense_target

   def _ammo_depleted_action(self, ship_state, game_state):
       """Out of bullets: if we have mines, drop one where it hits a cluster (driving to
       the densest one first if needed); if we're also out of mines, ram asteroids for
       collision hits. 
       
       Returns (thrust, turn, fire, drop_mine)."""
       asts = game_state["asteroids"]
       msize = game_state["map_size"]
       sx, sy = ship_state["position"]
       if ship_state["mines_remaining"] != 0:
           payoff = mine_blast_payoff(asts, sx, sy, 90, msize)
           endgame = (game_state["time_limit"] != math.inf
                      and game_state["time_limit"] * 30 - self.current_frame <= 90)
           # Drop right here if the blast would catch a worthwhile cluster (or time's up).
           if ship_state["can_deploy_mine"] and (payoff >= self.AMMO_MINE_MIN or (endgame and payoff >= 1)):
               self._say_mine(f"Out of bullets -> dropping a mine here; its blast will catch {payoff} asteroid(s).")
               return 0.0, 0.0, False, True
           # Otherwise drive to the densest cluster and mine it there.
           self._say("ammo_move", "Out of bullets -> driving to the densest asteroid cluster to mine it.")
           target = self._densest_point(asts, msize)
           if target is None:
               return 0.0, 0.0, False, False
           thrust, turn, _dist = navigate_to(ship_state, target[0], target[1], msize)
           return thrust, turn, False, False
       # No bullets AND no mines: ram the nearest asteroid -- a collision still scores a hit.
       self._say("ammo_ram", "Out of bullets and mines -> ramming asteroids; every crash still counts as a hit.")
       best, bestd = None, math.inf
       for a in asts:
           ax, ay = a["position"]
           d = (ax - sx) * (ax - sx) + (ay - sy) * (ay - sy)
           if d < bestd:
               bestd, best = d, a
       thrust, turn, _dist = navigate_to(ship_state, best["position"][0], best["position"][1], msize, stop=False)
       return thrust, turn, False, False

   def _ring_steal_action(self, ship_state, game_state):
       """Fly onto the ring-trapped opponent and mine them. Returns (thrust, turn, fire, drop_mine)."""
       msize = game_state["map_size"]
       can_mine = ship_state["can_deploy_mine"] and ship_state["mines_remaining"] != 0
       lives = ship_state["lives_remaining"]
       # Shoot the ring on the way in while we still have our starting invulnerability to
       # spend (we're aimed into the ring, so forward shots hit it). The instant we take
       # our first hit (deaths > 0) we go silent -- firing would ditch the respawn invuln
       # we now need to survive the dive.
       not_collided_yet = ship_state["deaths"] == 0
       fire = not_collided_yet
       if not_collided_yet:
           self._say("steal_fire", "Charging in -> firing forward into the ring while we still have full invuln to spend.")
       else:
           self._say("steal_holdfire", "Took our first hit -> holding fire to keep our invulnerability for the rest of the dive.")
       invuln = ship_state["respawn_time_left"]
       opp = find_opponent(game_state, ship_state["id"])
       if opp is None:
           # Opponent already destroyed -- mission accomplished. Spend our mines for points.
           if can_mine:
               self._say_mine("Opponent destroyed -> dumping our remaining mines for extra points.")
               return 0.0, 0.0, fire, True
           return 0.0, 0.0, fire, False
       tx, ty = opp["position"]
       thrust, turn, dist = navigate_to(ship_state, tx, ty, msize)
       self._say("steal_go", "Thrusting into the opponent's ring, aiming to stop right on top of them.")
       # Keep driving onto them until we're basically coincident. Movement is free during
       # invulnerability (it doesn't cancel it), so even after a hit resets our speed to 0
       # we burn back in and re-center instead of stalling short.
       if dist <= self.STEAL_CENTER_DIST:
           thrust = 0.0   # coincident -- hold right on top of them
           if not self._steal_started:
               self._steal_started = True
               self._say("steal_arrive", "Centered on the opponent -> now timing mines to our invulnerability.")
       drop_mine = False
       if lives <= 1:
           # Last life: no more respawns coming, so get the mines out now.
           if can_mine:
               drop_mine = True
               self._say_mine("On our last life -> dumping mines to take the opponent down with us.")
       elif can_mine and dist <= self.STEAL_MINE_DIST and invuln <= self.STEAL_MINE_INVULN:
           # Patient mining: we held off while invulnerable (staying alive + centered);
           # now the invuln is almost spent, so drop the mine at the last moment.
           drop_mine = True
           if invuln > 0.0:
               self._say_mine("Invulnerability almost out -> dropping a mine now to spend the full invuln window.")
           else:
               self._say_mine("On the opponent and vulnerable -> dropping a mine.")
       return thrust, turn, fire, drop_mine

   @property
   def name(self) -> str:
        """
        Simple property used for naming controllers such that it can be displayed in the graphics engine

        Returns:
            str: name of this controller
        """
        return "T.E. v2.0"

        

        
    