import pandas as pd
import rpy2.robjects as robjects

robjects.r['load']("https://github.com/vdorie/npci/blob/master/examples/ihdp_sim/data/ihdp.RData")
