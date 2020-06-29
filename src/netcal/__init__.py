# Copyright (C) 2019 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


# NOTE: We copy part of the netcal source code directly so that we can use pyodide. Netcal depends on torch
# and we also don't want to deal with other netcal's dependencies when we build the pyodide package.
# When and if we package kale as a separate library, independent of the pyodide packages and the game,
# this package will be removed in favor of including netcal as dependency


"""
API Reference
=============
This is the detailled API reference for the confidence calibration framework. This framework can be used to
obtain well-calibrated confidence estimates from biased estimators like Neural Networks.
The API reference contains a detailled description of all available methods and their parameters. For
miscellaneous examples on how to use these methods, see readme below.

Available packages

.. autosummary::
   :toctree: _autosummary

   binning
   scaling
   regularization
   metrics
   presentation


Each calibration method must inherit the base class :class:`AbstractCalibration`. If you want to write your own method and
include into the framework, include this class as the base class.

.. autosummary::
   :toctree: _autosummary_abstract_calibration
   :template: custom_class.rst

   AbstractCalibration

"""

name = 'calibration'
__version__ = '1.0'

from .Decorator import accepts, dimensions, global_accepts, global_dimensions
