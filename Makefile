train-MLP:
	python train_MLP.py
train-CNN:
	python train_CNN.py
convert-MLP:
	python convert_MLP.py
convert-CNN:
	python convert_CNN.py
env:
	conda env create -f environment.yml