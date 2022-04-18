# AI Implementation for StalkPrice
### Predict n days prices for an ETF using 2 datasets and 2 models
## How to use it?
**Clone this repo to your local machine and run the following commands**
```
git clone https://github.com/mhfong/fyp-ai.git
cd fyp-ai/
pip install -r requirements.txt
python main.py
```
**To change dataset type (default is qqq):**
```
python main.py -d top10
```
**To change predict days to 5 (default is 1):**
```
python main.py -p 5
```
**To skip training:**
```
python main.py -t
```
**To change model to GRU (default is lstm):**
```
python main.py -m gru
```
