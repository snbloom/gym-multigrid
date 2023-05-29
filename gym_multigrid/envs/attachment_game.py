from gym_multigrid.multigrid import *

class AttachmentActions:
    available=['still', 'left', 'right', 'forward', 'cry', 'interact']

    still = 0
    left = 1
    right = 2
    forward = 3
    # Express distress
    cry = 4
    # Comfort another agent
    interact = 5

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

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size,
            actions_set=AttachmentActions
        )


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        for number, index, reward in zip(self.num_balls, self.balls_index, self.balls_reward):
            for i in range(number):
                self.place_obj(Ball(self.world, index, reward))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)


    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        for j,a in enumerate(self.agents):
            if a.index==i or a.index==0:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if fwd_cell.index in [0, self.agents[i].index]:
                    fwd_cell.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    self._reward(i, rewards, fwd_cell.reward)

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    # copied from multigrid to edit for attachment  
    def step(self, actions):
        self.step_count += 1

        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        done = False

        for i in order:

            if self.agents[i].terminated or self.agents[i].paused or not self.agents[i].started or actions[i] == self.actions.still:
                continue

            # Get the position in front of the agent
            fwd_pos = self.agents[i].front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # Rotate left
            if actions[i] == self.actions.left:
                self.agents[i].dir -= 1
                if self.agents[i].dir < 0:
                    self.agents[i].dir += 4

            # Rotate right
            elif actions[i] == self.actions.right:
                self.agents[i].dir = (self.agents[i].dir + 1) % 4

            # Move forward
            elif actions[i] == self.actions.forward:
                if fwd_cell is not None:
                    if fwd_cell.type == 'goal':
                        done = True
                        self._reward(i, rewards, 1)
                    elif fwd_cell.type == 'switch':
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.grid.set(*self.agents[i].pos, None)
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            elif 'build' in self.actions.available and actions[i]==self.actions.build:
                self._handle_build(i, rewards, fwd_pos, fwd_cell)

            # Pick up an object
            elif actions[i] == self.actions.pickup:
                self._handle_pickup(i, rewards, fwd_pos, fwd_cell)

            # Drop an object
            elif actions[i] == self.actions.drop:
                self._handle_drop(i, rewards, fwd_pos, fwd_cell)

            # Toggle/activate an object
            elif actions[i] == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)

            # Done action (not used by default)
            elif actions[i] == self.actions.done:
                pass

            else:
                assert False, "unknown action"

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
        num_balls=[2],
        agents_index = [1,2],
        balls_index=[0],
        balls_reward=[1],
        zero_sum=True)

        

