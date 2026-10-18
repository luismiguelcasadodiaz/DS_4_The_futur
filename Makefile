# Set the default goal so running `make` with no arguments prints the help menu

.DEFAULT_GOAL := help
environment := DS4

.PHONY: help
help: ## Show this help menu
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
# ###############################################################33

.PHONY: DS4_ex00
DS4_ex00: ## Calculates the Confusoin matrix and plot it
	python ex00/Confusion_Matrix.py data/truth.txt data/predictions.txt

.PHONY: DS4_ex01
DS4_ex01: ## Shows correlation between the target and all the other features
	python ex01/Heatmap.py data/Train_knight.csv

.PHONY: DS4_ex02
DS4_ex02: ## Display 4 graphs
	python ex02/variances.py data/Train_knight.csv data/Test_knight.csv

.PHONY: DS4_ex03
DS4_ex03: ## Standarize, print and plot data
	python ex03/Feature_Selection.py data/Train_knight.csv


.PHONY: DS4_ex04
DS4_ex04: ## Normalize, print and plot data
	python ex04/Tree.py data/Train_knight.csv

.PHONY: DS4_ex05
DS4_ex05: ## splits Train_knight.csv into Training_knight.csv and Validation_knight.csv
	python ex05/KNN.py data/Train_knight.csv

.PHONY: DS4_ex06
DS4_ex06: ## splits Train_knight.csv into Training_knight.csv and Validation_knight.csv
	python ex05/democracy.py data/Train_knight.csv	
# ###############################################################33

.PHONY: set
set: ## Set a python environment for this project
	sh -c "python3 -m venv $(environment) && . $(environment)/bin/activate && pip install --upgrade pip &&pip install -r requirements.txt"

.PHONY: activate
activate: ## Activate the python environment for this project
	@echo "Run: source ../$(environment)/bin/activate"

.PHONY: deactivate
deactivate: ## Deactivate the python environment for this project
	deactivate

.PHONY: unset
unset: ## removes the python 
	rm -rf $(environment)

.PHONY: upgrade
upgrade: ## Upgrades pip
	pip install --upgrade pip

.PHONY: norminette
norminette: ## Run norminette on all .py files
	flake8  */*.py		

.PHONY: entrega
entrega: ## copies deliverable files to 42 repo
	mkdir ../../DS4/ex00
	mkdir ../../DS4/ex01	
	mkdir ../../DS4/ex02
	mkdir ../../DS4/ex03
	mkdir ../../DS4/ex04
	mkdir ../../DS4/ex05
	mkdir ../../DS4/ex06	
	mkdir ../../DS4/data
	cp data/Train_knight.csv ../../DS4/data/
	cp data/Test_knight.csv ../../DS4/data/
	cp ex00/Histogram.py ../../DS4/ex00/
	cp ex01/Correlation.py ../../DS4/ex01/
	cp ex02/points.py ../../DS4/ex02/
	cp ex03/standardization.py ../../DS4/ex03/
	cp ex04/Normalization.py ../../DS4/ex04/
	cp ex05/split.py ../../DS4/ex05/
	cp ex06/split.py ../../DS4/ex06/	
	cp Makefile ../../DS4
	cp requirements.txt ../../DS4

.PHONY: drop
drop: ## Drops items and data_202*_*** tables
	psql -U luicasad -d piscineds -h localhost -c "DROP TABLE IF EXISTS data_2022_oct; DROP TABLE IF EXISTS data_2022_nov; DROP TABLE IF EXISTS data_2022_dec; DROP TABLE IF EXISTS data_2023_jan; DROP TABLE IF EXISTS items;"

