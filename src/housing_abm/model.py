#top level Mesa model, monthly cycle
import yaml
import numpy as np
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from housing_abm.agents.renter import Renter

#class AtlantaHousingModel(Model):
 #   def __init__