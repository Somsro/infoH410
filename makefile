all:
	python setup.py build_ext --inplace


# Remove report files 
clean:
	python -c "from pathlib import Path; import shutil; report = Path('report'); [p.unlink() for p in report.glob('main.*') if p.name != 'main.tex' and p.is_file()]; shutil.rmtree('build', ignore_errors=True)"

test:
	python setup.py build_ext --inplace
	python test.py

qlearner:
	python game.py qlearner

expectimax:
	python game.py expectimax

deeplearning:
	python game.py deeplearning
