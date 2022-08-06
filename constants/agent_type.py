# importing enum for enumerations
import enum
 
# creating enumerations using class
class AgentType(enum.Enum):
    A2C = 1
    DDPG = 2
    DQN = 3
    MULTI_AGENT = 4
    