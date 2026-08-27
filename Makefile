.PHONY: test run compile
compile:
	python -m compileall rw_mc_studio app.py

test: compile
	python -m pytest -q

run:
	python -m streamlit run app.py
