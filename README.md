# Trustworthy and Explainable AI Project (2024-2025)

## Building a smaller, Dutch LM-Polygraph
The motivation behind our project was to implement a simple framework of Uncertainty Estimation (UE) methods for LLMs, similar to LM-Polygraph [[1]](#1). Our approach differs in that we only perform UE only on one task (Machine Translation), implement our own UE methods, and use a Dutch text quality generation metrics (BERTscore). 

Its main functionalities are creating a table of UE scores for each of the UE methods as well as computing the Prediction Rejection Ratio (PRR) for each of the UE methods. The PRR is a measure of how well a system can decide to reject a prediction based on the uncertainty score. 

## Setup Instructions

### Step 1: Install Required Libraries
1. Initialize and activate the virtual environment:
    ```python
   python -m venv env
   env/scripts/activate
   ```
2. Install dependencies from the requirements file:
   ```python
   pip install -r requirements.txt
   ```
### Step 2: Run the Main Script
To execute the project, run:
```python
python main.py
```
## Miscellanous
See the "figures" folder for the heatmap and Prediction Rejection Ratio plots

## Credits
The main methodology of our project was heavily inspired by the work done by Fadeeva et al. [[1]](#1) as we attempted to implement their framework on a Dutch language model. 

## References
<a id="1">[1]</a> 
Fadeeva et al. (2023). LM-Polygraph: Uncertainty estimation for language models. In Y. Feng & E. Lefever (Eds.), *Proceedings of the 2023 conference on empirical methods
in natural language processing: System demonstrations* (pp. 446–461). 