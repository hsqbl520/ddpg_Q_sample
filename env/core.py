import numpy as np

class EntityState:
    def __init__(self):
        self.p_pos = None

class AgentState(EntityState):
    def __init__(self):
        super().__init__()
        self.before_action_p_pos=None
        self.v = None
        self.yaw = None

class AgentAction:
    def __init__(self):
        self.u = None

class Entity:
    def __init__(self):
        self.size = 0.050
        self.state = EntityState()

class Landmark(Entity):
    def __init__(self):
        super().__init__()
        pass

class Obstacle(Entity):
    def __init__(self):
        super().__init__()
        pass

class Agent(Entity):
    def __init__(self):
        super().__init__()
        self.amax = 1.0
        self.wmax = 1.0
        self.state = AgentState()
        self.action = AgentAction()

class World:
    def __init__(self):
        self.agent = []
        self.landmark = []
        self.obstacle = []
        self.dim_p = 2
        self.dt = 0.1

    def step(self):
        control = self.agent.action.u
        self.integrate_state(control)

    def integrate_state(self,control):
        self.agent.state.before_action_p_pos=(self.agent.state.p_pos).copy()

        # control[0] = math.tanh(control[0])
        # control[1] = math.tanh(control[1])

        a_low = -self.agent.amax
        a_high = self.agent.amax
        a = a_low + (control[0] + 1) / 2 * (a_high - a_low)

        w_low = -self.agent.wmax
        w_high = self.agent.wmax
        w = w_low + (control[1] + 1) / 2 * (w_high - w_low)

        self.uav_model(self.agent, self.dt, a, w)

    def uav_model(self,agent,dt,a,w):
        #获取状态
        x0=agent.state.p_pos[0]
        y0=agent.state.p_pos[1]
        #v0 = agent.state.v
        #yaw0 = agent.state.yaw

        #运动学模型
        #xf = x0 + v0*dt*math.cos(yaw0 + w*dt/2)
        #yf = y0 + v0*dt*math.sin(yaw0 + w*dt/2)
        #vf = v0 + a*dt
        #yawf = yaw0 + w*dt
        xf = x0+a
        yf = y0 + w

        #更新状态
        agent.state.p_pos[0]=xf
        agent.state.p_pos[1]=yf
        #agent.state.v = vf
        #agent.state.yaw = yawf

