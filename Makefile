.PHONY: install data index test eval brief app lint clean all

install:
	pip install -e ".[dev,app]" tabulate

data:
	hrx data

index:
	hrx index

test:
	pytest -q

eval:
	hrx eval

brief:
	hrx brief "반도체사업본부 차장 승진 적체를 어떻게 해소해야 하나?" --stdout

app:
	streamlit run src/hrx/app/dashboard.py

lint:
	ruff check src tests

clean:
	rm -rf data/index data/synthetic reports .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

all: data index test eval
