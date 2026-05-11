all:
	python setup.py build_ext --inplace

test:
	python setup.py build_ext --inplace
	python test.py

qlearner:
	python game.py qlearner

expectimax:
	python game.py expectimax

deeplearning:
	python game.py deeplearning
