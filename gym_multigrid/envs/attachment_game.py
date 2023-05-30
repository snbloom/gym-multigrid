from gym_multigrid.multigrid import *
import random

LOVING_REWARD = 5
UNLOVING_REWARD = -5
NEW_OBJ_REWARD = 20
OLD_OBJ_REWARD = 1
CHILD_RADIUS = 4
CHILD_RADIUS_PENALTY = -1
COMPOUNDING_LOSS_THRESHOLD = -20
COMPOUNDING_LOSS_PENALTY = -1

class AttachmentActions:
    available=['still', 'left', 'right', 'forward', 'cry', 'play']

    still = 0
    left = 1
    right = 2
    forward = 3
    # Express distress
    cry = 4
    # # Comfort another agent (parent)
    # comfort = 5
    # play with an object (child)
    play = 5
    # # Parent agent collects a task object
    # pickup = 7


class ParentAgent(Agent):
    def __init__(self, world, index=0, view_size=7, p_respond=0.9, p_loving=0.9):
        super().__init__(world, index, view_size)

        self.p_respond = p_respond
        self.p_loving = p_loving


class ChildAgent(Agent):
    def __init__(self, world, index=0, view_size=7, p_respond=0.9, p_loving=0.9):
        super().__init__(world, index, view_size)

        self.cumulative_reward = 0

class AttachmentGame(MultiGridEnv):
    """
    Environment in which a parent and child agent interact
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        num_balls=[],
        agents_index = [],
        balls_index=[],
        balls_reward=[],
        zero_sum = False,
        view_size=7,

    ):
        self.num_balls = num_balls
        self.balls_index = balls_index
        self.balls_reward = balls_reward
        self.zero_sum = zero_sum

        self.world = World
        self.actions = AttachmentActions

        agents = []

        # Make child agent, view_size = 3 because that's the minimum allowed elsewhere in the code --> change this perhaps, if time
        agents.append(ChildAgent(self.world, 0, 3))

        # Make parent agent, view_size = whole grid
        agents.append(ParentAgent(self.world, 1, size+1, p_respond=1, p_loving=1))

        # shortcuts for parent and child agents
        self.child = agents[0]
        self.parent = agents[1]
        self.parent_objs = []
        self.child_objs = []
        self.child_visited_idxs = []

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size,
            actions_set=self.actions
        )


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        self.parent_objs = []
        self.child_objs = []

        for i, [number, index, reward] in enumerate(zip(self.num_balls, self.balls_index, self.balls_reward)):
            for _ in range(number):
                if i==1: self.parent_objs.append(self.place_obj(Ball(self.world, index, reward)).tolist())
                elif i==0: self.child_objs.append(self.place_obj(Ball(self.world, index, reward)).tolist())
                else: raise Exception("in _gen_grid: index not 0 or 1 or 2")  

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)


    # def _reward(self, i, rewards, reward=1):
    #     """
    #     Compute the reward to be given upon success
    #     """
    #     for j,a in enumerate(self.agents):
    #         if a.index==i or a.index==0:
    #             rewards[j]+=reward
    #         if self.zero_sum:
    #             if a.index!=i or a.index==0:
    #                 rewards[j] -= reward

    # def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
    #     if fwd_cell:
    #         if fwd_cell.can_pickup():
    #             if fwd_cell.index in [0, self.agents[i].index]:
    #                 fwd_cell.cur_pos = np.array([-1, -1])
    #                 self.grid.set(*fwd_pos, None)
    #                 self._reward(i, rewards, fwd_cell.reward)

    # def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
    #     pass


    def get_distance(self, sq1, sq2):
        x1, y1 = sq1
        x2, y2 = sq2
        return ((x2-x1)**2 + (y2-y1)**2) ** 0.5


    def parent_step(self, child_action, rewards):
        # Get the position in front of the parent
        fwd_pos = self.parent.front_pos
        cur_pos = self.parent.pos
        child_pos = self.child.pos

        # Get the contents of the cell in front of the parent
        fwd_cell = self.grid.get(*fwd_pos)

        # print(self.grid.grid)
        # print("self.parent_objs", self.parent_objs)
        # print("self.child_obs", self.child_objs)

        dists = [[idx, self.get_distance(cur_pos, ball_pos)] for (idx, ball_pos) in enumerate(self.parent_objs)]
        # print(dists)
        min_dist = min(dists, key=lambda pair: pair[1])
        # print(min_dist)

        # if child is crying
        if child_action == self.actions.cry and random.random() < self.parent.p_respond:
            print("parent responds to child crying")
            # if fwd square is the child, comfort the child
            if fwd_cell is not None and fwd_cell.type == "agent":
                # flip on p_loving to get loving versus not loving response
                if random.random() < self.parent.p_loving: rewards[0] += LOVING_REWARD   # loving
                else: rewards[0] += UNLOVING_REWARD                                      # not loving
                print("parent comforting")
            # else if moving forward decreases the distance between parent and child do that
            elif self.get_distance(fwd_pos, child_pos) < self.get_distance(cur_pos, child_pos): 
                self.grid.set(*fwd_pos, self.parent)
                self.grid.set(*self.parent.pos, None)
                self.parent.pos = fwd_pos
            # else if child is to the right, turn to the right
            elif cur_pos[0] < child_pos[0]: self.parent.dir = 0
            # else if child is to the left, turn to the left
            elif cur_pos[0] > child_pos[0]: self.parent.dir = 2
            # else if child is below, face down 
            elif cur_pos[1] < child_pos[1]: self.parent.dir = 1
            # else if child is above, face upwards 
            elif cur_pos[1] > child_pos[1]: self.parent.dir = 3
            # this should cover all the bases so comment an error if not
            else: raise Exception("something weird happened... :P")
            
        # else locate nearest parent object and move towards that
        else:
            dists = [[idx, self.get_distance(cur_pos, ball_pos)] for (idx, ball_pos) in enumerate(self.parent_objs)]
            # print("dists", dists)
            closest_ball = self.parent_objs[min(dists, key=lambda pair: pair[1])[0]]
            # print("cur_pos", cur_pos,"closest_ball", closest_ball)

            # if the fwd_cell is the closest ball or moving forward would bring closer to closest ball then move forward
            if self.get_distance(fwd_pos, closest_ball) < self.get_distance(cur_pos, closest_ball) and not (fwd_cell is not None and fwd_cell.type == "agent"):
                self.grid.set(*fwd_pos, self.parent)
                self.grid.set(*self.parent.pos, None)
                self.parent.pos = fwd_pos
            else: 
                # if closest_ball is to the right, turn to the right
                if cur_pos[0] < closest_ball[0]: self.parent.dir = 0
                # else if closest_ball is to the left, turn to the left
                elif cur_pos[0] > closest_ball[0]: self.parent.dir = 2
                # else if closest_ball is below, face down 
                elif cur_pos[1] < closest_ball[1]: self.parent.dir = 1
                # else if closest_ball is above, face upwards 
                elif cur_pos[1] > closest_ball[1]: self.parent.dir = 3

            if self.get_distance(cur_pos, closest_ball) == 0: 
                # pop off old ball
                self.parent_objs.remove([closest_ball[0], closest_ball[1]])
                # generate new ball?
                self.parent_objs.append(self.place_obj(Ball(self.world, 1, 1)).tolist())
            
                

    def child_step(self, action, rewards):
        if self.child.terminated or self.child.paused or not self.child.started or action == self.actions.still:
            return

        # Get the position in front of the agent
        fwd_pos = self.child.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.child.dir -= 1
            if self.child.dir < 0:
                self.child.dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.child.dir = (self.child.dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.grid.set(*fwd_pos, self.child)
                self.grid.set(*self.child.pos, None)
                self.child.pos = fwd_pos
            # self._handle_special_moves(0, rewards, fwd_pos, fwd_cell)

        # Signal distress
        elif action == self.actions.cry:
            print("CHILD IS CRYING")
            # change color to display to user --> except idk how
            # other reward handled in parent step function

        # Play with an object
        elif action == self.actions.play:
            if fwd_cell is not None:
                if fwd_cell.type == 'objgoal':
                    rewards[0] += NEW_OBJ_REWARD if fwd_pos not in self.child_visited_idxs else OLD_OBJ_REWARD
                    if fwd_pos not in self.child_visited_idxs: self.child_visited_idxs.append(fwd_pos)

        # # Pick up an object
        # elif action == self.actions.pickup:
        #     # self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
        #     pass

        # # Drop an object
        # elif action == self.actions.drop:
        #     self._handle_drop(i, rewards, fwd_pos, fwd_cell)

        # # Toggle/activate an object
        # elif action == self.actions.toggle:
        #     if fwd_cell:
        #         fwd_cell.toggle(self, fwd_pos)

        # # Done action (not used by default)
        # elif action == self.actions.done:
        #     pass
        else:
            assert False, "unknown action"

        # Handle far from parent penalty
        parent_pos = self.parent.pos
        child_pos = self.child.pos
        if self.get_distance(parent_pos, child_pos) > CHILD_RADIUS: rewards[0] += CHILD_RADIUS_PENALTY

        # Handle cumulative penalty
        if self.child.cumulative_reward < COMPOUNDING_LOSS_THRESHOLD: rewards[0] += COMPOUNDING_LOSS_PENALTY

        # Update cumulative reward
        self.child.cumulative_reward += rewards[0]
        print("cumulative_reward", self.child.cumulative_reward)

    # copied from multigrid to edit for attachment  
    def step(self, actions):
        self.step_count += 1

        rewards = np.zeros(len(actions))
        done = False

        self.child_step(actions[0], rewards)
        self.parent_step(actions[0], rewards)

        if self.step_count >= self.max_steps:
            done = True

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            obs = [self.grid.encode_for_agents(self.agents[i].pos) for i in range(len(actions))]

        obs=[self.objects.normalize_obs*ob for ob in obs]

        return obs, rewards, done, {}


class AttachmentGame4HEnv10x10N2(AttachmentGame):
    def __init__(self):
        super().__init__(size=10,
        num_balls=[3,3],
        agents_index = [0,1],
        balls_index=[0,1],
        balls_reward=[1,1],
        zero_sum=False)

        

