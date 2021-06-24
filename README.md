# Kyle - a Calibration Toolkit

## Note:
This library is currently in the alpha stage and breaking changes can happen at any time. Some
central features are currently missing and will be added soon.

## Overview
This library contains utils for measuring and visualizing calibration of probabilistic classifiers as well as for 
recalibrating them. Currently, only methods for recalibration through post-processing are supported, although we plan
to include calibration specific training algorithms as well in the future.

Kyle is model agnostic, any probabilistic classifier can be wrapped with a thin wrapper called `CalibratableModel` which
supports multiple calibration algorithms. For a quick intro overview of the API have a look at the calibration demo 
notebook (the notebook with executed cells can be found in the docu).

Apart from tools for analysing models, kyle also offers support for developing and testing custom calibration metrics
and algorithms. In order not to have to rely on evaluation data sets and trained models for delivering labels and confidence 
vectors, with kyle custom samplers based on fake classifiers can be constructed. A note explaining the
theory behind fake classifiers will be published soon.
These samplers can
also be fit on some data set in case you want to mimic it. Using the fake classifiers, an arbitrary number of ground 
truth labels and miscalibrated confidence vectors can be generated to help you analyse your algorithms (common use cases
will be analysis of variance and bias of calibration metrics and benchmarking of recalibration algorithms).


Currently, several algorithms in kyle use the [calibration framework library](https://github.com/fabiankueppers/calibration-framework) under the hood although this is subject 
to change.

## Installation
Kyle can be installed from pypi, e.g. with
```
pip install kyle-calibration
```