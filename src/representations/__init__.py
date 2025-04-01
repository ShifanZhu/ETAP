from .event_stack import MixedDensityEventStack
from .voxel_grid import VoxelGrid

class EventRepresentationFactory:
    @staticmethod
    def create(representation_config):
        config = representation_config.copy()
        representation_name = config.pop('representation_name')

        if representation_name == 'event_stack':
            return MixedDensityEventStack(**config)
        elif representation_name == 'voxel_grid':
            return VoxelGrid(**config)
        else:
            raise ValueError("Unsupported representation_name.")