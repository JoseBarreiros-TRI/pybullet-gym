import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import pdb
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from . import kuka
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class KukaReachEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps=1000,
               reward_type='cost_visak',
               debug = False):
    #print("KukaGymEnv __init__")
    self._isDiscrete = isDiscrete
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40
    self.velocities_n1 = np.zeros(7)
    self.reward_type= reward_type
    self.debug=debug

    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid < 0):
        cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    else:
      p.connect(p.DIRECT)
    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    self.seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())

    if self.debug:
      print("observationDim: ", observationDim)

    observation_high = np.array([largeValObservation] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
      action_dim = 3
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None

  def reset(self):

    #print("KukaGymEnv _reset")
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    # p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
    self.tableUid=p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -0.64,
               0.000000, 0.000000, 0.0, 1.0)
    self.target_xyz = np.array([0.5,0,0.5])
    p.setGravity(0, 0, -10)
    self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, debug=self.debug)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    self._observation = self._kuka.getObservation()

    # Cartesian position of end effector in world frame.
    gripper_state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    EE_xyz = gripper_state[0]
    #orn = state[1]
    #euler = p.getEulerFromQuaternion(orn)

    if self.debug:
      print("EE_xyz: {}".format(EE_xyz))
    self._observation.extend(list(EE_xyz))
    #pdb.set_trace()

    return self._observation

  def step(self, action):
    # actions are end effector dx, dy, dz
    if (self._isDiscrete):
      dv = 0.005
      dx = [0, -dv, dv, 0, 0, 0, 0][action]
      dy = [0, 0, 0, -dv, dv, 0, 0][action]
      dz = [0, 0, 0, 0, 0, -dv, dv][action]
      realAction = [dx, dy, dz]
    else:
      #print("action[0]=", str(action[0]))
      dv = 0.005
      dx = action[0] * dv
      dy = action[1] * dv
      dz = action[2] * dv
      realAction = [dx, dy, dz]
    return self.step2(realAction)

  def step2(self, action):
    for i in range(self._actionRepeat):
      self._kuka.applyAction(action)
      p.stepSimulation()
      if self._termination():
        break
      self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
      self._observation = self.getExtendedObservation()

    done = self._termination()
    # npaction = np.array([
    #     action[3]
    # ])  #only penalize rotation until learning works well [action[0],action[1],action[3]])
    # actionCost = np.linalg.norm(npaction) * 10.
    #print("actionCost")
    #print(actionCost)
    reward = self._reward(self.reward_type) #- actionCost
    #print("reward")
    #print(reward)

    #print("len=%r" % len(self._observation))

    return np.array(self._observation), reward, done, {}

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

    base_pos, orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                              height=RENDER_HEIGHT,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
    #renderer=self._p.ER_TINY_RENDERER)

    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    #print (self._kuka.endEffectorPos[2])
    EE_state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = EE_state[0]

    # Time limit termination
    if (self.terminated or self._envStepCounter > self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True

    # Termination due to collision with table
    maxDist = 0.005
    closestPoints = p.getClosestPoints(self.tableUid, self._kuka.kukaUid, maxDist)
    if (len(closestPoints)):  #(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1
      return True

    # Success.
    success_threshold = 0.005
    if np.linalg.norm(self.target_xyz - actualEndEffectorPos) < success_threshold:
      self.terminated = 1
      return True


    return False

  def _reward(self, reward_type):

    EE_state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = EE_state[0]
    vector_to_goal = self.target_xyz - actualEndEffectorPos

    # robot state
    positions=[]
    velocities=[]
    for joint in range(7):
      joint_state=p.getJointState(self._kuka.kukaUid,joint)
      positions.append(joint_state[0])
      velocities.append(joint_state[1])

    iiwa_velocities = np.array(velocities)
    iiwa_acceleration_estimated = (iiwa_velocities - self.velocities_n1)/(self._timeStep*self._actionRepeat)
    self.velocities_n1 = iiwa_velocities

    #pdb.set_trace()
    if reward_type=="cost_visak":
      ALPHA = 10
      BETA = 0.1
      GAMMA = 0.001
      cost_goal = np.power(
          np.linalg.norm(vector_to_goal),
          2)
      cost_ = (np.exp(-ALPHA*cost_goal) -
            BETA * (np.linalg.norm(iiwa_velocities)) -
            GAMMA * (np.linalg.norm(iiwa_acceleration_estimated)))
      reward = cost_
    elif reward_type=="cost_goal":
      # Distance to goal.
      cost_goal = vector_to_goal.dot(vector_to_goal)
      reward = 10*cost_goal
    elif reward_type=="success":
      success_threshold = 0.006
      if np.linalg.norm(vector_to_goal) < success_threshold:
        reward=1
      else:
        reward=0

    if self.debug:
      print("d_to_goal: {}, q_dot: {}, q_ddot: {}".format(np.linalg.norm(vector_to_goal),iiwa_velocities, iiwa_acceleration_estimated))
      print("reward: {}".format(reward))

    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
