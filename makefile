all:
	python setup.py build_ext --inplace

clean:
	python -c "from pathlib import Path; import shutil; report = Path('report'); extensions = ['.log', '.aux', '.bbl', '.blg', '.fdb_latexmk', '.fls', '.synctex.gz', '.out', '.toc']; [p.unlink() for p in report.glob('*') if p.is_file() and (p.suffix in extensions or any(p.name.endswith(e) for e in extensions))]; [p.unlink() for p in report.glob('main.*') if p.name != 'main.tex' and p.is_file()]; shutil.rmtree('build', ignore_errors=True)"

expectimax:
	python game.py expectimax

td:
	python game.py td

dql:
	python game.py dql

testExpectimax:
	python test.py expectimax

testTD:
	python test.py td

testDQL:
	python test.py dql
