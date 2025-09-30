#########################################################################
# Change parameters here to experiment with different settings.
# This file is used to configure parameters for data loading, preprocessing,
# model training, and evaluation.
#########################################################################

DATAFILE = "income_census.csv"
CORRELATIONFILE = "income_correlation_matrix.csv"
HEATMAPFILE = "income_correlation_matrix_heatmap.png"
COUNTRYTHRESHOLD = 100
CORRELATIONTHRESHOLD = 0.85
TESTSIZE = 0.2
RANDOMSTATE = 42
CLASSWEIGHT = None # Options: None, 'balanced', or a dict like {0: 1, 1: 3}
MAXITER = 1000
SCALER = "standard" # Options: 'standard', 'minmax01', 'minmax-11'
SHOWHEATMAP = False # Options: True, False
