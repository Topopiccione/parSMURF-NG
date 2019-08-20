### parSMURFng
### 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
import os
import sys
import json

import numpy as np
from skopt import Optimizer
from skopt.space import Integer
from skopt.space import Real

def importJsonCfg(cfgFilename):
	with open(cfgFilename) as f:
		data = json.load(f)
	return data["BOptParams"]

def unpackVariables(params):
	BO_params = dict()
	BO_vars = []
	var_idx = 0
	var_names = dict()
	fixed_vars = dict()

	for variable in params["Variables"]:
		if variable["min"] != variable["max"]:
			if variable["Type"] == "Real":
				BO_vars.append(Real(variable["min"], variable["max"], name=variable["Name"]))
			if variable["Type"] == "Integer":
				BO_vars.append(Integer(variable["min"], variable["max"], name=variable["Name"]))
			var_names[variable["Name"]] = var_idx
			var_idx += 1
		else:
			fixed_vars[variable["Name"]] = variable["min"]

	BO_params["Variables"] = BO_vars
	BO_params["VariableNames"] = var_names
	BO_params["FixedVars"] = fixed_vars

	return BO_params

def importFromFile(varNames, fixedVars):
	# line in file are defined as:
	# nParts fp ratio k numTrees mtry auprc/Pending
	names = ['nParts', 'fp', 'ratio', 'k', 'numTrees', 'mtry']
	pointLst = []
	auprcs = []
	with open("tempOpt.txt", "r") as f:
		for line in f:
			spltLine = line.split(sep=' ')
			newPoint = []
			for id in range(6):
				if names[id] in fixedVars:
					continue
				newPoint.append(int(spltLine[id]))
			pointLst.append(newPoint)
			if spltLine[6] is 'P':
				print("Point already pending. Quitting the optimizer...")
				exit(0)
			auprcs.append(float(spltLine[6]))
	return pointLst, auprcs

def convertPoint(pt, varNames, fixedVars):
	newPt = []
	names = ['nParts', 'fp', 'ratio', 'k', 'numTrees', 'mtry']
	for vName in names:
		if vName in fixedVars:
			newPt.append(int(fixedVars[vName]))
		else:
			newPt.append(int(pt[varNames[vName]]))
	newPt.append('P')
	return newPt

def updateFile(pt):
	with open("tempOpt.txt", "a") as fout:
		fout.write(" ".join([str(x) for x in pt]))

def tellHimToStopGoddamit():
	with open("tempOpt.txt", "a") as fout:
		fout.write("DONE")

def optimize(cfgFilename):
	params = importJsonCfg(cfgFilename)
	BO_params = unpackVariables(params)

	varSpace = BO_params["Variables"]
	opt = Optimizer(varSpace, base_estimator=params["BaseEstimator"], acq_func=params["AcquisitionFunction"],
		acq_optimizer=params["AcquisitionOptimizer"])

	optMaxIter = int(params["maxOptIter"])

	if os.path.isfile("tempOpt.txt"):
		pointLst, auprcs = importFromFile(BO_params["VariableNames"], BO_params["FixedVars"])
		for i in range(0, len(auprcs)):
			opt.tell(pointLst[i], auprcs[i])

	if len(pointLst) > optMaxIter:
		tellHimToStopGoddamit()

	pt = opt.ask()
	pt = convertPoint(pt, BO_params["VariableNames"], BO_params["FixedVars"])
	updateFile(pt)


if __name__ == "__main__":
	cfgFilename = sys.argv[1]
	optimize(cfgFilename)
