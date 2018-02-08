# Supplement to Farmer et al (2018).

## Contents:

1. [Solution Speciation](http://nbviewer.jupyter.org/github/oscarbranson/Farmer_2018_Supplement/blob/master/Solution%20Speciation.ipynb): An investigation of different solution chemistry calculation methods (various databases in PHREEQC, and MyAMI).
2. [Model_Fitting_Bootstrap](http://nbviewer.jupyter.org/github/oscarbranson/Farmer_2018_Supplement/blob/master/Model_Fitting_Bootstrap.ipynb): Fitting the model to data, calculating the model 95% confidence interval by bootstrapping, and investigating the possible influence of isotope fractionation during incorporation.
3. [Omega - 1](http://nbviewer.jupyter.org/github/oscarbranson/Farmer_2018_Supplement/blob/master/Omega%20-%201.ipynb): Re-calculating all available paired B/Ca and d11B data from inorganic calcites to a uniform 'precipitation rate' scale, and fitting the model.

## Requirements:

All the above work in Python 3.6. Code in these notebooks relies on various functions contained in the inorg_b module, in this repository. Speciation calculation also requires a working installation of [phreeqpy](http://www.phreeqpy.com/).