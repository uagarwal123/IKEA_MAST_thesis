# agent-evaluation-framework

This repository contains the code for our bachelor thesis, in which we develop an evaluation framework for multi-agent systems. 

## Setup

```bash
pip install -r requirements.txt
```

## Data

The parsers and notebooks expect the MAD dataset at `data/MAST-Data/`:

```
data/
  MAST-Data/
    MAD_full_dataset.json
    MAD_human_labelled_dataset.json
```

## Running the parsers

Each parser is a standalone script. Run from the repo root:

```bash
python parsers/ag2_parser/ag2_parser.py
python parsers/metagpt_parser/metagpt_parser.py
python parsers/chatdev_parser/chatdev_parser.py
# etc.
```

Output is written as JSON next to each parser, e.g. `parsers/ag2_parser/ag2_output_mad.json`.

## Repository structure

```
parsers/        # one parser per agent framework
data_understanding/  # exploratory notebooks and analysis
data/           # MAD dataset 
```
