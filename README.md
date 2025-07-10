# Student Academic Performance Dashboard

## Introduction

This project is part of the OMSA Practicum. The objective is to build a real-time dashboard integrating assessment data to provide up-to-date snapshots of student performance relative to their historical trends. The system also includes a hybrid recommendation engine using FastAPI for the backend and React for the frontend.

## Folder Structure

```
PracticumSummer2025
├── backend                # FastAPI app and business logic
│   ├── app               # Main FastAPI app
│   ├── core              # Core utilities and settings
│   ├── models            # Data models
│   ├── services          # Service layer
│   └── venv              # Virtual environment
├── data
│   ├── raw               # Raw input data
│   └── processed         # Cleaned and transformed datasets
├── frontend              # React app with Bootstrap
├── models                # ML model training and evaluation
├── notebooks             # EDA and experimentation
├── streamlit             # Streamlit prototype dashboard
├── .gitignore
├── README.md
├── setup.bat             # Windows setup script
```

## Requirements

### Backend (FastAPI)

- Python 3.9+
- pip
- `venv` or `virtualenv`
- Dependencies listed in `backend/requirements.txt`

### Frontend (React + Vite)

- Node.js 16+
- npm or yarn
- Modern web browser (Chrome, Firefox, Edge)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/bvo0293/PracticumSummer2025.git
cd PracticumSummer2025
```

### 2. Set Up Python Environment

```bash
# Windows
python -m venv backend/venv
backend\venv\Scripts\activate
pip install -r backend/requirements.txt
```

```bash
# macOS/Linux
python3 -m venv backend/venv
source backend/venv/bin/activate
pip install -r backend/requirements.txt
```

### 3. Add Raw Data Files

Place the following `.csv` files into the `data/raw/` folder:

```
Courses.csv
GraduationAreaSummary.csv
IlluminateData2022.csv
IlluminateData2023.csv
IlluminateData2024.csv
IlluminateData2025.csv
Student Teacher Grade 2022.csv
Student Teacher Grade 2023.csv
Student Teacher Grade 2024.csv
Student Teacher Grade 2025.csv
```

### 4. Preprocess Data

```bash
# Windows
python data\DataPreProcessing.py
python data\Illuminate\ ETL.py
```

```bash
# macOS/Linux
python3 data/DataPreProcessing.py
python3 data/Illuminate\ ETL.py
```

### 5. Train Machine Learning Model

```bash
# Windows
python models\prepare_data.py
python models\GBM_ML.py
```

```bash
# macOS/Linux
python3 models/prepare_data.py
python3 models/GBM_ML.py
```

### 6. Frontend Setup (React + Bootstrap)

#### Install Node.js (if not installed)

##### Windows:

1. Visit [https://nodejs.org](https://nodejs.org)
2. Download and install the **LTS version**
3. Follow installation steps (ensure npm is selected)

Verify installation:

```bash
node -v
npm -v
```

##### macOS:

```bash
brew install node
```

#### Install Frontend Dependencies

```bash
deactivate  # Exit Python venv if active
cd frontend
npm install
npm install bootstrap bootstrap-icons
cd ..
```

### 7. Run the Application

```bash
# Windows
setup.bat
```

```bash
# macOS/Linux
./setup.sh
```

### Optional: We only have the demo for LLM in streamlit, so you can try to run this
Install streamlit
```bash
pip3 install streamlit
```
Re-activate virtual environment
```bash
source backend/venv/bin/activate
```
Run streamlit 
```bash
streamlit run streamlit/stream_lit_with_LLM.py
```

## Contributors

- Bao Vo
- Hui Li
- Duc H Nguyen

---


