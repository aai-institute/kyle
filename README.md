# Kyle - a Calibration Toolkit

This library contains utils for measuring and visualizing calibration of probabilistic classifiers as well as for 
recalibrating them. Currently, only methods for recalibration through post-processing are supported, although we plan
to include calibration specific training algorithms as well in the future.

Kyle is model agnostic, any probabilistic classifier can be wrapped with a thin wrapper called `CalibratableModel` which
supports multiple calibration algorithms. For a quick intro overview of the API have a look at the calibration demo 
notebook (the notebook with executed cells can be found in the docu).

Apart from tools for analysing models, kyle also offers support for developing and testing custom calibration metrics
and algorithms. In order not to have to rely on evaluation data sets and trained models for delivering labels and confidence 
vectors, with kyle custom samplers based on [fake classifiers](our paper/review) can be constructed. These samplers can
also be fit on some data set in case you want to mimic it. Using the fake classifiers, an arbitrary number of ground 
truth labels and miscalibrated confidence vectors can be generated to help you analyse your algorithms (common use cases
will be analysis of variance and bias of calibration metrics and benchmarking of recalibration algorithms). Several
pre-configured fake classifiers mimicking common models, e.g. vision models trained on MNIST and CIFAR10, are implemented
in kyle and can be used out of the box. 
