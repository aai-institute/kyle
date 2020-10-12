from netcal.metrics import MCE as netcal_MCE

from metrics.base_calibration_error import BaseCalibrationError


class MCE(BaseCalibrationError, netcal_MCE):
    def __init__(self):
        super().__init__()
