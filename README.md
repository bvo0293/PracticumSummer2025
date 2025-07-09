# Student Academic Performance Dashboard

## Introduction
This project is a part of OMSA. The objective is to build a dashboard to integrate real-time assessment data to provide an up-to-the-minute snapshot of a student's performance relative to their historical average and create a hybrid recommendation system (using FastAPI as the backend and React as the frontend).

## Folder Structure
The folder structure of the project is as follows:

```bash
├── PracticumSummer2025 
    ├── backend (FastAPI)
    │   ├── app (contains main.py file) 
    │   ├── core
    │   ├── models
    │   ├── services
    │   └── venv
    ├── data
    │   ├── raw  (raw data collected)
    │   └── processed (data after being)
    ├── frontend (React + Booststrap)
    ├── models (for ML practice)
    ├── notebooks (for EDA and other discoveries)
    ├── streamlit (for streamlit prototype)
    ├── .gitignore
    ├── README.md
    ├── setup.bat
```

## Requirements

### Backend (FastAPI)

- Python 3.9+ (recommended)
- pip (Python package manager)
- Virtual environment tool (`venv` or `virtualenv`)
- Dependencies specified in `requirements.txt`

### Frontend (React + Vite)

- Node.js 16+ (recommended)
- npm (comes with Node.js) or yarn
- Modern web browser (Chrome, Firefox, Edge, etc.)

## How to Use
Please follow the step one by one:

### 1. Clone the Repository

```bash
git clone https://github.com/bvo0293/PracticumSummer2025.git
cd PracticumSummer2025
```

### 2. Create a Python virtual environment and install dependencies

```bash
# On Windows
python -m venv backend/venv
.\backend\venv\Scripts\activate
pip install -r backend/requirements.txt
```
```bash
# On macOS/Linux
python3 -m venv backend/venv
source backend/venv/bin/activate

pip install -r backend/requirements.txt
```

### 3. Place the necessary data files into the data/raw folder (please let them all be in .csv format)

```bash
Courses.csv
GraduationAreaSummary.csv
IlluminateData2022.csv
IlluminateData2023.csv
IlluminateData2024.csv
IlluminateData2025.csv
Student Teacher Grade 2022.csv
Student Teacher Grade 2023.csv
Student Teacher Grade 2024.csv
Student Teacher Grade 2022.csv
```
### 4. Make sure the files are all available then derive processed files
The DataPreProcessing.py file performs transformations (combining Student Teacher Grade data, filtering for only High School student, determining active students) to create the Final_DF.csv in `data/processed/` folder.
The Illuminate ETL.py file performs transformation to combine the Illuminate data to a unique file IlluminateCombined.csv in `data/processed/` folder.
```bash
# On Windows
python .\data\DataPreProcessing.py
python .\data\Illuminate ETL.py
```
```bash
# On macOS/Linux
python3 "./data/DataPreProcessing.py"
python3 "./data/Illuminate ETL.py"
```
### 5. Train model
The prepare_data.py helps prep the data in the way that would allows us to utilize different features to run our GBM model in GBM_ML.py
The GBM_ML.py generates the necessary ML model and results 
```bash
# On Windows
python .\models\prepare_data.py
python .\models\GBM_ML.py 
```
```bash
# On macOS/Linux
python3 "./models/prepare_data.py"
python3 "./models/GBM_ML.py"
```

### 6. Install React + Boostrap and their dependencies
If Node.js and npm (Node Package Manager) are not installed on your system
For Windows:
	1.	Go to the official Node.js download page
👉 https://nodejs.org
	2.	Download the LTS (Long Term Support) version
This version is more stable and recommended for most users.
	3.	Run the installer (.msi file)
	•	Double-click the downloaded file.
	•	Accept the license agreement.
	•	Choose the destination folder (default is fine).
	•	Make sure npm package manager is selected (it is by default).
	•	Optionally install tools for native modules (optional, but useful for some packages).
	4.	Finish the installation and restart your terminal/PowerShell.

For MacOS:
```bash
brew install node
```
Then, run
```bash
deactivate
cd frontend
npm install
npm install bootstrap
npm install bootstrap-icons
cd ..
```

### 7. Run the program
```bash
# On Windows
setup.bat
```
```bash
# On macOS/Linux
./setup.sh
```


## Contributors
Vo, Bao; Li, Hui; Nguyen, Duc H
