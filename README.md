# Predicting n days High & Low price of QQQ using different models
  **Download this repo to your local machine and run the following commands**
  ```
  pip install -r requirements.txt
  python main.py
  ```
  **To change dataset (default is 1):**
  ```
  python main.py -d 2
  ```
  **To change predict days to 5 (default is 1):**
  ```
  python main.py -p 5
  ```
  **To skip training (default is 1):**
  ```
  python main.py -t 0
  ```
  **To change model to LSTM (default is GRU):**
  ```
  python main.py -m lstm
  ```
