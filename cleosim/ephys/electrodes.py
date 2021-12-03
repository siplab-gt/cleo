from cleosim import Recorder
# probe coords

# array coords

# signal
class Signal:
    pass

# electrode group
class ElectrodeGroup(Recorder):
    def __init__(self, name, coords, signals=[]):
        super().__init__(name)

