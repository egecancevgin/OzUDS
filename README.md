# OzU CS552 Project

## Files and Folders Structure:
- `main.py`: Machine Learning Operation happens in this file, after the normalization phases in utility_functions.py.
- `requirements.txt`: Required Python packages and their versions.
- `utility_function.py`: Utility functions for data normalization, took over 30 hours to handle but they are done in Turkish. Also they need other datasets that I currently cannot provide because it is such a hassle to turn them all to turkish and encrypt them :).
- `datasets`: Contains the dataset `env.csv`, this dataset is a fake but the structure is a bit similar, much complicated and full of unique values though. Had 93% accuracy with the real dataset.
- `app/`: App folder consists of FastAPI model deployments.  
- `app/main.py`: Model deployment file that uses FastAPI Framework.
- `app/templates/index.html`: Web page front-end HTML file.
- `app/static/styles.css`: Web page front-end CSS file.
- `.pkl` files are for model and encoder object to inverse and make prediction.

 
## For Linux
In Debian systems, let's first update the apt package manager and install 'pip', the Python package installer.
``` bash
apt update
apt install python3-pip
```

Then, install the packages with their specific versions listed in the requirements.txt file. The content of the file should look like this:
``` py
pandas==2.2.2
openpyxl==3.1.5
scikit-learn==1.5.1
paramiko==3.4.0
...
```

Now, to install the packages, enter the following command in the terminal:
``` bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

To train the model, run the main.py file in the current directory:
``` bash
python3 main.py
```

You will see two different pickle files, one of them is the labelencoder object, other one is for the trained model. Send these pickle objects to the ./app folder with cp command or use ctrl-c ctrl-v from the computer:
```
cp *.pkl ./app/
```


To launch the FastAPI application, first go to the ./app folder and run the command below:
``` sh
python -m uvicorn app.main:app --reload
```


## For Windows:

After Python installation, ensure pip is up to date. Open Command Prompt and run:
``` powershell
python -m pip install --upgrade pip
```

Then, install the packages with their specific versions listed in the requirements.txt file. The content of the file should look like this:
``` powershell
pandas==2.2.2
openpyxl==3.1.5
scikit-learn==1.5.1
paramiko==3.4.0
...
```

Now, to install the packages, enter the following command in the Command Prompt:
``` powershell
pip install -r requirements.txt
```

To train the model, run the main.py file in the current directory by entering:
``` powershell
python main.py
```

You will see two different pickle files. One of them is the label encoder object, and the other is for the trained model. Move these pickle objects to the ./app folder manually using copy-paste or run the following in Command Prompt:
``` powershell
move *.pkl .\app\
```

To launch the FastAPI application, first navigate to the ./app folder in Command Prompt:
``` powershell
cd app
python -m uvicorn app.main:app --reload
```


