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
python -m venv backend/venv
.\backend\venv\Scripts\activate
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

```bash
python .\data\DataPreProcessing.py
python '.\data\Illuminate ETL.py'
```
### 5. Train model
```bash
python .\models\prepare_data.py
python .\models\GBM_ML.py 
```

### 6. Install React + Boostrap and their dependencies
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
setup.bat
```

## Contributors
Vo, Bao; Li, Hui; Nguyen, Duc H