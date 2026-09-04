# RAGAS Evaluation Toolkit

This folder contains utility scripts used to benchmark Reactome RAG pipelines with Ragas.

- `test_generator.py` synthesizes question/answer test sets from the example corpora. It uses the Ragas `TestsetGenerator` to create LangChain-based evaluation datasets.
- `evaluator.py` runs either the basic or advanced Reactome RAG chain over a test set and scores the outputs with Ragas metrics (answer relevancy, context utilization, faithfulness, context recall), saving both responses and evaluation reports.

## Requirements

- Python 3.12 (project default) with Poetry environment
- `ragas` (see poetry.lock for the pinned version)
- OpenAI access: `OPENAI_API_KEY` (and optional Azure configuration if required)
- An installed embeddings bundle (`./bin/embeddings_manager install ...`).
  `evaluator.py` defaults to whichever bundle is active; override with
  `--embeddings-dir`.

Run `poetry install` to set up dependencies, then activate the virtual environment via `poetry shell` or use `poetry run` for individual commands.

## Usage

1. **Generate test sets**

   ```bash
   poetry run python src/evaluation/test_generator.py \
     --path src/evaluation/example \
     --model gpt-4o-mini \
     --temperature 0.3 \
     --test_size 10 \
     --distributions simple=0.25 reasoning=0.25 multi_context=0.25 conditional=0.25
   ```

   Outputs are stored in a `testsets/` directory of your choosing.

2. **Evaluate a RAG configuration**

   ```bash
   poetry run python src/evaluation/evaluator.py \
     --testset_dir <your testset dir> \
     --rag_type advanced \
     --model gpt-4o-mini
   ```

   Responses and metric reports are written to `response/<rag_type>/` and `evals/<rag_type>/` inside the testset directory.

Adjust the paths, model names, and distribution weights as needed for local experimentation.

