from netcal.metrics import ACE as netcal_ACE

from metrics.base_calibration_error import BaseCalibrationError


class ACE(BaseCalibrationError, netcal_ACE):
    def __init__(self):
        super().__init__()
