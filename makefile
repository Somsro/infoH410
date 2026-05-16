all:
	python setup.py build_ext --inplace

clean:
	python -c "from pathlib import Path; import shutil; report = Path('report'); [p.unlink() for p in report.glob('main.*') if p.name != 'main.tex' and p.is_file()]; shutil.rmtree('build', ignore_errors=True)"

expectimax:
	python game.py expectimax

td:
	python game.py td

dqn:
	python game.py dqn

testExpectimax:
	python test.py expectimax

testTD:
	python test.py td

testDQN:
	python test.py dqn

stats:
	python stats.py
