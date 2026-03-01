effectuer un "pip install -r requirements.txt"

https://www.kaggle.com/datasets/kartik2112/fraud-detection/data?select=fraudTrain.csv
telecharger le dataset "fraudTrain.csv"  et placer au même niveau que le pip

puis placer dans directory au même niveau que EDA

placez vous dans le dossier api :

pip install uvicorn --user

python -m uvicorn main:app --reload

http://127.0.0.1:8000/docs#/
