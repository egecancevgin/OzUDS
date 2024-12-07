# OzUDS
Ozyegin CS552


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


For Windows:



