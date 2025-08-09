from .feature_tracking_online import FeatureTrackingDataModule
from .e2d2 import E2d2DataModule
from .penn_aviary import PennAviaryDataModule
from .event_kubric import EventKubricDataModule
from .evimo2 import Evimo2DataModule

class DataModuleFactory:
    @staticmethod
    def create(data_config):
        dataset_name = data_config['dataset_name']

        if dataset_name == 'feature_tracking_online':
            return FeatureTrackingDataModule(**data_config)
        elif dataset_name == 'e2d2':
            return E2d2DataModule(**data_config)
        elif dataset_name == 'penn_aviary':
            return PennAviaryDataModule(**data_config)
        elif dataset_name == 'event_kubric':
            return EventKubricDataModule(**data_config)
        elif dataset_name == 'evimo2':
            return Evimo2DataModule(**data_config)
        elif dataset_name == 'cear':
            return FeatureTrackingDataModule(**data_config)
        else:
            raise ValueError("Unsupported dataset_name.")
