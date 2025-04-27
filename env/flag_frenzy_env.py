import numpy as np
from pathlib import Path
from gymnasium import Env, spaces
import random
import copy

import env.SimulationInterface as SimulationInterface
from env.SimulationInterface import EntitySpawned, Victory, AdversaryContact
from env.SimulationInterface import (
    Entity, ControllableEntity, Unit, Squadron, Package, VictoryConditions, TargetGroup, Flag
)
from env.SimulationInterface import Faction, SimulationData, Vector3, Formation, EntityDomain, ControllableEntityManouver
from env.SimulationInterface import PlayerEvent_Move, PlayerEvent_CAP, PlayerEvent_RTB, PlayerEvent_Commit

import logging

from datetime import datetime
import os
import random

class FlagFrenzyEnv(Env):
    def __init__(self, save_replay=True, dynasty_engages=False):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.save_replay = save_replay
        self.dynasty_engages = dynasty_engages

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("FlagFrenzyEnv initialized")

        self.simulation_event_handlers = {
            EntitySpawned: self._entity_spawned,
            Victory: self._victory_event,
            AdversaryContact: self._adversary_contact
        }

        self.info = {}

        self.area_min = np.array([-336122.0, -998052.0])
        self.area_max = np.array([ 743277.0,  278723.0])

        self.frame_rate = 600
        self.simulation_json = (Path(__file__).parent / "Simulation.json").resolve().read_text()
        self.num_entity_features = 26
        self.param_vector_size = 10
        self.mission_vector_size = 7

        self.max_game_time = 9000 # soft upper bound for time (2.5 hours)

        self.max_entities = 100
        self.max_priority = 1 # This may change in the future

        self.max_ammo = 50
        self.max_velocity = 1029.0  # 3x 343 m/s
        self.map_bound = 1300000  # -1300000 to 1300000 map bounds

        self.controllable_entity_ids = {11, 35} # TODO: do not hardcode!

        SimulationInterface.Initialize()

        self.factions = [Faction.Legacy.value, Faction.Dynasty.value] # Legacy is 0, Dynasty is 1
        self.logger.info(f"factions {self.factions}")
        self.setup_simulation()

        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(4), # Representing the 4 main actions
            # TODO: Can use 7 or 8, check squad ID + 6 spline point coordinates
            "params": spaces.Box(low=-1.0, high=1.0, shape=(self.param_vector_size,), dtype=np.float32) # Fixed size vector for action parameters
        })

        self.reward = {          # default values
            "time_step"          : -0.0005,
            "idle_step"          : -0.005,
            "engage_step"        :  0.08,
            "flagship_engage"    :  0.20,
            "enemy_kill"         :  0.10,
            "friendly_loss"      : -0.20,
            "flagship_destroy"   :  0.50,
            "b1_alive_step"      :  0.001,
            "b1_safe_step"       :  0.02,
            "b1_danger_step"     : -0.01,
            "b1_death"           : -0.50,
            "move_oob"           : -0.02,
            "bad_rtb"            : -0.05,
        }

        # self.observation_space = spaces.Tuple(( # Unbounded obs space range
        #     # E
        #     spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_entities, self.num_entity_features), dtype=np.float32),  # Dimension fo feature vector

        #     # V
        #     spaces.Tuple((  # Assuming 1000 max entity count
        #         spaces.MultiBinary(self.max_entities),  # Legacy radar visibility
        #         spaces.MultiBinary(self.max_entities)   # Dynasty radar visibility
        #     )),

        #     # M
        #     spaces.Box(low=0.0, high=1.0, shape=(self.mission_vector_size,), dtype=np.float32) # Size of mission status vector
        # ))

        self.observation_space = spaces.Dict({
            "controllable_entities": spaces.MultiBinary(self.max_entities),
            "entities": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_entities, self.num_entity_features),
                dtype=np.float32
            ),
            "visibility": spaces.Dict({
                "legacy": spaces.MultiBinary(self.max_entities),
                "dynasty": spaces.MultiBinary(self.max_entities)
            }),
            "mission": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.mission_vector_size,),
                dtype=np.float32
            ),
            # For action masking:
            "valid_engage_mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_entities * self.max_entities,),
                dtype=np.float32
            ),
            "entity_id_list": spaces.Box( # TODO: Debug
                low=0.0, high=1000,
                shape=(len(self.entity_id_list),),
                dtype=np.float32
            ),
        })
        
        self.logger.info("Flag Frenzy env created")

    def setup_simulation(self):
        self.entities = {}
        self.entity_id_list = [] # Separate list to keep track of stable ordered entity IDs
        self.id_to_index = {}
        self.target_groups = []
        self.victory_conditions = {a: None for a in self.factions}
        self.flags = {a: None for a in self.factions}
        self.FrameIndex = 0
        self.sim_data = None

        self._enemy_killed_ids = set()
        self._friendly_killed_ids = set()
        self._flagship_destroyed_rewarded = False
        self._b1_dead_penalized = False

        self.total_episode_reward = 0

        self.step_counter = 0
        
        self.red_win = False
        self.blue_win = False

        self.blue_engaged_in_combat = False
        self.flagship_hit = False

        # Set up the simulation from json
        self.simulation_handle = SimulationInterface.CreateSimulationFromData(self.simulation_json, True)

        # Perform a pre-tick so that all initial data is initialized
        events = SimulationInterface.TickPreSimulation(self.simulation_handle, SimulationData())
        self.process_simulation_events(events)

        # Set up the Simulation data for the next frame
        self.sim_data = SimulationData()



    def shuffle_entity_indices(self):
        self.logger.info(f"[DEBUG] Shuffling entity_id_list! FrameIndex={self.FrameIndex}, Step={self.step_counter}")

        random.shuffle(self.entity_id_list)

        self.id_to_index = {eid: idx for idx, eid in enumerate(self.entity_id_list)}

    def reset(self, seed=None, options=None):
        # Destroy previous data
        if hasattr(self, 'simulation_handle'):

            if self.save_replay:
                replay = SimulationInterface.ExportJSON(self.simulation_handle)

                datetime_now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

                replays_dir = "Replays"

                if not os.path.exists(replays_dir):
                    os.makedirs(replays_dir)

                replay_file_name = f"{replays_dir}/{datetime_now}.json"

                f = open(replay_file_name, "w")
                f.write(replay)
                f.close()

            SimulationInterface.DestroySimulation(self.simulation_handle)

        self.setup_simulation()

        self.logger.info("Resetting the env....")
        self.shuffle_entity_indices() # Add shuffling so model is not dependent on same order of entities

        self.controllable_entities = np.zeros(self.max_entities, dtype=np.int8)
        for eid in self.controllable_entity_ids:
            if eid in self.id_to_index:
                self.controllable_entities[self.id_to_index[eid]] = 1

        self.logger.info("Env reset")
        self.current_entity_id_list = copy.deepcopy(self.entity_id_list)

        obs = self._get_observations()

        self.info = {
            "num_entities": len(self.entity_id_list),
            "valid_engage_mask": self.get_valid_engage_mask().flatten().astype(np.float32).tolist(),
            "frame_index": self.FrameIndex,
        }


        self.info["entity_id_list"] = self.current_entity_id_list

        return obs, self.info

    def _tick(self):
        events = SimulationInterface.TickSimulation(self.simulation_handle, self.sim_data, self.frame_rate)
        self.sim_data = SimulationData()
        self.process_simulation_events(events)

    # This calls event handlers
    def process_simulation_events(self, events):
        for event in events:
            handler = self.simulation_event_handlers.get(type(event))

            if handler:
                handler(event)
            else:
                # print(f"Frame {self.FrameIndex}: Unhandled {Event}")
                pass

    def _entity_spawned(self, event):
        if issubclass(event.Entity.__class__, ControllableEntity):
            self._controllable_entity_spawned(event)
        elif type(event.Entity) == TargetGroup:
            self._target_group_spawned(event)
        elif type(event.Entity) == VictoryConditions:
            self._set_victory_conditions(event)
        elif type(event.Entity) == Flag:
            self._flag_spawned(event)
        else:
            pass # We don't care about others for now

    def _controllable_entity_spawned(self, Event):
        if Event.Entity.HasParent():
            return
        Identifier = Event.Entity.Identifier

        # self.logger.info(f"Controllable {type(Event.Entity).__name__} spawned with identifier {Identifier} at pos {Event.Entity.Pos.x, Event.Entity.Pos.y, Event.Entity.Pos.z}")

        self.entities[Event.Entity.EntityId] = Event.Entity
        self.entity_id_list.append(Event.Entity.EntityId)
        self.id_to_index[Event.Entity.EntityId] = len(self.entity_id_list) - 1

    def _victory_event(self, Event):
        # self.logger.info(f"Victory condition triggered for {Event.VictorFaction}. Score = {Event.Score}")
        pass

    def _adversary_contact(self, Event):
        if not self.dynasty_engages:
            return

        if Event.Entity.Faction.value == 1:
            if Event.TargetGroup.GetDomain().value & Event.Entity.GetTargetDomains():
                if random.random() > 0.1:   # 90% chance of engagement
                    self._engage_target_action(Event.Entity, Event.TargetGroup, 0, 0)

    def _target_group_spawned(self, Event):
        # self.logger.info(f"TargetGroup spawned for {Event.Entity.Faction}")
        self.target_groups.append(Event.Entity)

    def _set_victory_conditions(self, Event):
        # self.logger.info(f"Set victory conditions for {Event.Entity.Faction}") 
        self.victory_conditions[Event.Entity.Faction.value] = Event.Entity

    def _flag_spawned(self, Event):
        # self.logger.info(f"Flag spawned for {Event.Entity.Faction}")
        self.flags[Event.Entity.Faction.value] = Event.Entity

    def step(self, action):

        self.flagship_hit = False
        self.blue_engaged_in_combat = False

        self.execute_action((action["action_type"], action["params"]))
        self.info = {}

        self._tick()
        self.step_counter += 1

        self.FrameIndex += self.frame_rate

        obs = self._get_observations()

        terminal_reward = self.get_terminal_reward()
        game_rewards = self.compute_reward()
        rewards = terminal_reward + game_rewards

        self.total_episode_reward += rewards

        engage_mask = self.get_valid_engage_mask()
        num_entities = len(self.entity_id_list)
        engage_mask_flat = engage_mask.flatten().astype(np.float32)

        terminated = terminal_reward != 0
        truncated = self.get_time_elapsed() >= self.max_game_time

        if terminated or truncated:
            self.logger.info(f"Episode ended. Reward: {rewards}, Total Reward: {self.total_episode_reward}, Length: {self.step_counter}")

        # Ensure no NANs
        assert np.isfinite(rewards), f"Reward is not finite: {rewards}"
        for k, v in obs.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    assert np.all(np.isfinite(vv)), f"Obs[{k}][{kk}] has NaNs or Infs"
            else:
                assert np.all(np.isfinite(v)), f"Obs[{k}] has NaNs or Infs"


        self.info.update({
            "frame_index": self.FrameIndex,
            "num_entities": len(self.entities),
            "reward": rewards,
            "mission_status": self.get_mission_status_vector().tolist(),
            "valid_engage_mask": engage_mask_flat.tolist(),  # for RLlib
            "num_entities": num_entities,
            "episode_len": self.step_counter,
            "termination_cause": (
                "blue_win" if self.blue_win
                else "red_win" if self.red_win
                else "none"
            ),
            "entity_id_list": obs["entity_id_list"],    
            "total_episode_reward": self.total_episode_reward,  
        })

        return obs, rewards, terminated, truncated, self.info

    def compute_reward(self):
        reward = 0.0
        R = self.reward

        # Time penalty
        reward += R["time_step"]

        # Engagement reward
        if self.blue_engaged_in_combat:
            reward += R["engage_step"]

        # Engage flagship
        if self.flagship_hit:
            reward += R["flagship_engage"]

        # Minimize friendly losses
        for eid, ent in self.entities.items():
            if eid not in self._friendly_killed_ids and not ent.IsAlive():
                if ent.Faction.value == 0: # Legacy
                    reward += R["friendly_loss"]
                    self._friendly_killed_ids.add(eid)

        # Reward for enemy kills
        for eid, ent in self.entities.items():
            if eid not in self._enemy_killed_ids and not ent.IsAlive():
                if ent.Faction.value == 1: # Dynasty
                    reward += R["enemy_kill"]
                    self._enemy_killed_ids.add(eid)

        # Reward for destroying flagship
        flagship = self.find_entity_by_name("Renhai")
        if flagship and not flagship.IsAlive() and not self._flagship_destroyed_rewarded:
            reward += R["flagship_destroy"]
            self._flagship_destroyed_rewarded = True

        # Reward for keeping B1 alive
        b1 = self.find_entity_by_name("B1")
        if b1 and b1.IsAlive():
            reward += R["b1_alive_step"]
            # Keep controllable Legacy close to B1
            min_dist = min(
                np.linalg.norm([b1.Pos.x - self.entities[eid].Pos.x,
                                b1.Pos.y - self.entities[eid].Pos.y])
                for eid in self.controllable_entity_ids
            )
            if min_dist <= 120_000: # B1 distance thresholds
                reward += R["b1_safe_step"]
            elif min_dist > 150_000:
                reward += R["b1_danger_step"]
        else:
            # One shot penalty for B1 dead
            if not self._b1_dead_penalized:
                reward += R["b1_death"]
                self._b1_dead_penalized = True

        # Penalty for movement out of bounds
        reward += self.info.pop("reward_move_oob", 0.0)

        # Penalty for unnecessary RTB
        reward += self.info.pop("reward_bad_rtb", 0.0)

        # Idle penalty
        idle = any(
        e.EntityId in self.controllable_entity_ids and
        e.CurrentManouver == ControllableEntityManouver.NoManouver
        for e in self.entities.values()
        )
        if idle:
            reward += R["idle_step"]

        self.info.update({
            "reward_time_penalty"   : R["time_step"],
            "reward_engage"  : R["engage_step"]   if self.blue_engaged_in_combat else 0.0,
            "reward_flagship_engage"    : R["flagship_engage"] if self.flagship_hit else 0.0,
            "reward_enemy_kill"     : R["enemy_kill"] * len(self._enemy_killed_ids),
            "reward_flag_kill"      : R["flagship_destroy"] if self._flagship_destroyed_rewarded else 0.0,
            "reward_friendly_loss"  : R["friendly_loss"] * len(self._friendly_killed_ids),
            "reward_move_oob"       : self.info.get("reward_move_oob", 0.0),
            "reward_bad_rtb"        : self.info.get("reward_bad_rtb", 0.0),
            "reward_idle"           : R["idle_step"] if idle else 0.0,
        })

        return reward

    def get_terminal_reward(self):
        """
        Computes the reward based on the mission status vector M.
        Mission status vector (M) format:
        [Flag Alive, B1 Alive, AWACS Alive, MADDOG Alive, B1 Returned, MADDOG Returned, Time]
        Blue team wins if:
            - Flagship is neutralized (Flag Alive == 0),
            - B1 is alive,
            - AWACS is alive,
            - Both B1 and MADDOG have returned (B1 Returned == 1 and MADDOG Returned == 1).
        Red team wins if either B1 or AWACS is destroyed.
        """
        M = self.get_mission_status_vector()
        flag_alive, b1_alive, awacs_alive, maddog_alive, b1_returned, maddog_returned, _ = M

        # Blue wins
        if flag_alive == 0 and b1_alive == 1 and awacs_alive == 1 and b1_returned == 1: # and maddog_returned == 1:
            self.blue_win = True
            return 1.0

        # Red wins
        if b1_alive == 0 or awacs_alive == 0:
            self.red_win = True
            return -1.0
        
        return 0.0
    
    def _get_observations(self):
        """Create state space matrix consisting of entity features, visibility, and mission status (E, V, M)."""
        entity_features = np.zeros((self.max_entities, self.num_entity_features), dtype=np.float32)

        for i, entity_id in enumerate(self.entity_id_list):
            entity = self.entities[entity_id]

            # Entity ID
            # entity_id = entity.EntityId 
            entity_index = i
            # Position (x, y)
            pos = entity.Pos
            # Velocity (vx, vy)
            vel = entity.Vel
            # Team ID one-hot (Legacy=0=Blue, Dynasty=1=Red)
            team_one_hot = [1, 0] if int(entity.Faction) == 0 else [0, 1]
            # Mission priority
            priority_raw = float(self.victory_conditions[entity.Faction.value].GetMissionEntityPriority(entity))
            priority = np.clip(priority_raw / self.max_priority, 0.0, 1.0)
            # Domain (land / air / sea)
            domain_value = entity.GetDomain().value
            if isinstance(domain_value, int):
                domain_type = [1 if domain_value == j else 0 for j in range(3)]
            # Target domain (land / air / sea) # Bit field (multiple can be set to true depending on active target)
            """ enum class SimProjectileTargetDomain : uint8_t {
                None = 0,
                Land = 1 << 0,
                Air = 1 << 1,
                Sea = 1 << 2,
            }; """
            target_domain = entity.GetTargetDomains()
            if isinstance(target_domain, int):
                target_domain_type = [(target_domain >> i) & 1 for j in range(3)]
            else:
                target_domain_type = [0, 0, 0]

            # Ammo remaining
            ammo_remaining = np.clip(entity.GetAmmo() / self.max_ammo, 0.0, 1.0)
            # Health (Binary) 
            is_alive = entity.IsAlive()
            # Reached base (Binary)
            reached_base = int(entity.HasReachedBase)
            
            # Combat Engagement Features
            # Has entity committed an action
            is_committing = int(entity.CurrentManouver == ControllableEntityManouver.Combat)
            # Engagement level (one-hot)
            engagement_level = entity.GetEngagementLevel()
            engagement_levels = ["defensive", "cautious", "assertive", "offensive"]
            engagement_one_hot = [1 if engagement_level == lvl else 0 for lvl in engagement_levels]
            # Weapons usage mode (one-hot)
            weapons_mode = entity.GetWeaponsUsageMode()
            weapons_modes = ["tight", "selective", "free"]
            weapons_one_hot = [1 if weapons_mode == mode else 0 for mode in weapons_modes]

            # Target Entity ID
            target = entity.GetTargetEntity()
            if target and target.EntityId in self.id_to_index:
                target_index = self.id_to_index[target.EntityId]
            else:
                target_index = -1  # -1 for no target

            features = np.array([
                pos.x / self.map_bound, 
                pos.y / self.map_bound, 
                vel.x / self.max_velocity, 
                vel.y / self.max_velocity,
                *team_one_hot,
                entity_index, priority,
                *domain_type,
                *target_domain_type,
                ammo_remaining,
                is_alive,
                reached_base,
                is_committing,
                *engagement_one_hot,
                *weapons_one_hot,
                target_index
            ], dtype=np.float32)

            entity_features[i] = features

        visible_to_legacy, visible_to_dynasty = self.get_radar_visibility()
        mission_status = np.array(self.get_mission_status_vector(), dtype=np.float32)
        entity_id_list_obs = np.array(copy.deepcopy(self.entity_id_list), dtype=np.float32)


        engage_mask_flat = self.get_valid_engage_mask().flatten().astype(np.float32)

        obs = {
            "controllable_entities": self.controllable_entities,
            "entities": entity_features,
            "visibility": {
                "legacy": visible_to_legacy,
                "dynasty": visible_to_dynasty
            },
            "mission": mission_status,
            "valid_engage_mask": engage_mask_flat,
            "entity_id_list": entity_id_list_obs
        }

        return obs
    
    def get_radar_visibility(self):
        visible_to_legacy = np.zeros(self.max_entities, dtype=np.int8)
        visible_to_dynasty = np.zeros(self.max_entities, dtype=np.int8)

        for target_group in self.target_groups:
            for entity in target_group.GetDetectedEntities():
                entity_id = entity.EntityId
                # if entity_id not in self.id_to_index:
                #     continue  # Entity not tracked, ignore

                idx = self.id_to_index[entity_id]
                if entity.Faction.value == 0:
                    visible_to_dynasty[idx] = 1 # seen by dynasty
                else:
                    visible_to_legacy[idx] = 1 # seen by legacy
        return visible_to_legacy, visible_to_dynasty


    def find_entity_by_name(self, name):
        for i, e in self.entities.items():
            if e.Identifier == name:
                return e
        return None

    def get_mission_status_vector(self):
        """Return mission status vector:
        (Flag Alive, B1 Alive, AWACS Alive, MADDOG Alive, B1 Returned, MADDOG Returned, Time)
        """

        flagship = self.find_entity_by_name("Renhai")
        b1 = self.find_entity_by_name("B1")
        awacs = self.find_entity_by_name("AWACS")
        maddog = self.find_entity_by_name("Maddog")

        flag_alive = int(flagship.IsAlive()) if flagship else 0
        b1_alive = int(b1.IsAlive()) if b1 else 0
        awacs_alive = int(awacs.IsAlive()) if awacs else 0
        maddog_alive = int(maddog.IsAlive()) if maddog else 0

        b1_returned = int(b1.HasReachedBase) if b1 else 0
        maddog_returned = int(maddog.HasReachedBase) if maddog else 0

        time_elapsed = self.get_time_elapsed() / self.max_game_time

        return np.array([flag_alive, b1_alive, awacs_alive, maddog_alive, b1_returned, maddog_returned, time_elapsed], dtype=np.float32)

    def get_time_elapsed(self):
        return self.FrameIndex / 60 # seconds

    def _decode_entity(self, val):
        # idx = int(np.clip(val, 0.0, 1.0) * len(self.entity_id_list))
        idx = int(np.clip(val, 0.0, 1.0) * (len(self.current_entity_id_list) - 1))

        idx = min(idx, len(self.entity_id_list) - 1)
        entity_id = self.current_entity_id_list[idx]
        # self.logger.info(f"Decoded entity index {idx} to entity ID {entity_id}")
        return entity_id

    def execute_action(self, action):
        """Dispatch and execute action in the environment."""
        a_d, params = action

        entity_id = self._decode_entity(params[0])


        if entity_id not in self.controllable_entity_ids: # Force no-op if not blue faction
            # self.logger.info(f"[ENV SKIP] Entity {entity_id} is not allowed to take actions.")
            return

        self.info["engage_action_taken"] = False

        if a_d == 0:
            return  # No-op

        elif a_d == 1:
            # Move Action: uses entity_id and 3 spline points (x, y)

            def to_map_coord(val):
                return np.clip(val, -1.0, 1.0) * self.map_bound

            entity_pos = self.entities[entity_id].Pos

            spline_xy = [
                (entity_pos.x, entity_pos.y),
                (to_map_coord(params[1]), to_map_coord(params[2])),
                (to_map_coord(params[3]), to_map_coord(params[4])),
                (to_map_coord(params[5]), to_map_coord(params[6])),
            ]

            for (x, y) in spline_xy:
                if (x < self.area_min[0] or x > self.area_max[0] or
                    y < self.area_min[1] or y > self.area_max[1]):
                    self.info["reward_move_oob"] = -0.02 # Set penalty for moving out of bounds


            spline = [Vector3([x, y, entity_pos.z]) for (x, y) in spline_xy]
            self.move_action(entity_id, spline)

        elif a_d == 2:
            # Return to Base Action
            # params[0]: entity_id (0.0 - 1.0)

            self.return_to_base_action(entity_id) # sending command to simulation

        elif a_d == 3:
            # Engage Target Action
            # params[0]: entity_id (0.0 - 1.0)
            # params[1]: target_id (0.0 - 1.0)
            # params[3]: engagement level (0.0 - 1.0) → [0, 3]
            # params[4]: weapons mode (0.0 - 1.0) → [0, 2]
            target_id = self._decode_entity(params[1])

            self.info["engage_action_taken"] = True
            self.info["engage_entity_id"] = entity_id
            self.info["engage_target_id"] = target_id

            engagement_level = int(np.clip(params[2], 0.0, 1.0) * 4) # mapping [0.0, 1.0] --> {0, 1, 2, 3}
            engagement_level = min(engagement_level, 3)

            weapons_mode = int(np.clip(params[3], 0.0, 1.0) * 3) # mapping [0.0, 1.0] --> {0, 1, 2}
            weapons_mode = min(weapons_mode, 2)


            self.engage_target_action(entity_id, target_id, engagement_level, weapons_mode) # sending command to simulation

        else:
            raise ValueError(f"Invalid action type {a_d}")

    # This is where the action functions map to events in the simulation   
    def move_action(self, entity_id, spline):
        move = PlayerEvent_Move()
        move.Entity = self.entities[entity_id]
        move.SplinePoints = spline # spline: list of points
        move.Throttle = 1.0
        move.Formation = Formation.Wall # Keep this constant for now

        self.sim_data.AddPlayerEvent(move)

    def return_to_base_action(self, entity_id):
        entity = self.entities[entity_id]
        if entity.GetAmmo() / self.max_ammo >= 0.5: # Penalize for RTB with > 50 ammo
            self.info["reward_bad_rtb"] = -0.05

        rtb = PlayerEvent_RTB()
        rtb.Entity = self.entities[entity_id]
        rtb.SplinePoints = [ rtb.Entity.Pos, self.flags[rtb.Entity.Faction.value].Pos]
        self.sim_data.AddPlayerEvent(rtb)

    def engage_target_action(self, entity_id, target_id, engagement_level, weapons_mode):
        if entity_id == target_id: # TODO: Resolve this
            # self.logger.info(f"[ENV WARN] Target {target_id} can not engage with itself.")
            return  # Don't engage

        entity = self.entities[entity_id]
        target_group = self.entities[target_id].GetTargetGroup()

        if target_group is None: # TODO: This should not be coming up during training!
            # self.logger.info(f"[ENV WARN] Target {target_id} has no valid target group, skipping engagement. {entity_id} targeting {target_id}")
            return  # Don't engage

        if self._engage_target_action(entity, target_group, engagement_level, weapons_mode):
            self.logger.info(f"Combat engagement! Entity {entity_id} engaged with target {target_id}")

            self.blue_engaged_in_combat = True
            flagship_id = self.find_entity_by_name("Renhai").EntityId
            if target_id == flagship_id:
                self.flagship_hit = True

    def _engage_target_action(self, entity, target_group, engagement_level, weapons_mode):
        selected_weapons = entity.SelectWeapons(target_group, False)
        if len(selected_weapons) == 0: # Domain mismatch, can't engage
            # self.logger.info(f"[ENV WARN] Target {target_id} does not match weapons domain.")
            return False

        def get_shoot_distance():
            shoot_distance = 0

            for key, weapon in selected_weapons.items():
                if weapon.Projectile.Range > shoot_distance:
                    shoot_distance = weapon.Projectile.Range

            return shoot_distance

        commit = PlayerEvent_Commit()
        commit.Entity = entity
        commit.TargetGroup = target_group
        commit.ManouverData.Throttle = 1.0
        commit.ManouverData.Engagement = engagement_level  # Should match enum or int mapping
        commit.ManouverData.WeaponUsage = weapons_mode         # Same here
        commit.ManouverData.Weapons = selected_weapons.keys()   # We just select all available weapons for now.
        commit.ManouverData.ShootDistance = get_shoot_distance()
        self.sim_data.AddPlayerEvent(commit)

        return True

    def get_valid_engage_mask(self):
        mask = np.zeros((self.max_entities, self.max_entities), dtype=np.int8)

        visible_to_legacy, visible_to_dynasty = self.get_radar_visibility()
        for src_id in self.current_entity_id_list:
            if src_id not in self.controllable_entity_ids:
                continue

            src = self.entities[src_id]
            src_idx = self.id_to_index[src_id]

            if not src.IsAlive(): # Make sure src entity is alive
                continue

            for tgt_id in self.current_entity_id_list:
                tgt = self.entities[tgt_id]
                tgt_idx = self.id_to_index[tgt_id]

                if not tgt.IsAlive(): # Make sure target entity is alive
                    continue
                if src.Faction == tgt.Faction: # Make sure src & target are different teams
                    continue

                if int(src.Faction) == 0 and visible_to_legacy[tgt_idx] == 0: # Make sure visibile in the radar
                    continue

                if tgt.GetTargetGroup() is None:
                    continue

                mask[src_idx, tgt_idx] = 1

        return mask


# # Added for simulation testing purposes
# if __name__ == "__main__":
#     env = FlagFrenzyEnv()

#     while env.compute_reward() == 0:
#         env._tick()
#         env._get_observations()

#         # Example on how to perform an attack
#         flagship = env.find_entity_by_name("Renhai")
#         b1 = env.find_entity_by_name("B1")

#         if (flagship.IsAlive()):
#             if b1.CurrentManouver != ControllableEntityManouver.Combat:
#                 env.execute_action([3, [b1.EntityId / env.max_entities, flagship.EntityId / env.max_entities, 0.0, 0.0]])
#         elif b1.CurrentManouver == ControllableEntityManouver.NoManouver:
#             env.execute_action([2, [b1.EntityId / env.max_entities]])

# # Added for env testing purposes
# if __name__ == "__main__":
#     env = FlagFrenzyEnv()
#     obs, info = env.reset()

#     for i in range(50):
#         action = env.action_space.sample()
#         print(f"Action sampled from the env: {action}")
#         observation, reward, term, trunc, info = env.step(action)
#         print(f"Reward at this step: {reward}")

#         # env.engage_target_action(11, 35, 0, 0)

#     env.close()
