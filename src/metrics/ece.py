from netcal.metrics import ECE as netcal_ECE

from metrics.base_calibration_error import BaseCalibrationError


class ECE(netcal_ECE, BaseCalibrationError):
    def __init__(self):
        super().__init__()
