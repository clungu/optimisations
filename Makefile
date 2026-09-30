.PHONY: export test docs dist

export:
	nbdev-export

test: export
	python -m unittest discover -s tests
	nbdev-test

docs:
	nbdev-docs
	nbdev-readme

dist: export
	python -m pip wheel --no-deps -w dist .
