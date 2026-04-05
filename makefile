all:
	python setup.py build_ext --inplace

test:
	python setup.py build_ext --inplace
	python test.py

play:
	python game.py
