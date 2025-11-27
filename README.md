# End-to-End Railway Delay Prediction & Data Warehousing Project

![System Architecture](project%20architecture.jpeg)

## Project Overview

This project delivers a complete **Data Engineering + Data Science
pipeline** designed to analyze historical railway operations and
**predict future delay risks**.\
It integrates **Python ETL**, **Machine Learning**, **SQL Server Data
Warehousing**, and a **Power BI Dashboard**, simulating a real
enterprise workflow.

The system enables a shift from **Reactive Analysis** (what happened?)
to **Proactive Decision Making** (what will happen?).

------------------------------------------------------------------------

## 📂 Repository Structure

    ├── 📁 Data/                     
    │   ├── railway.csv
    │   ├── railway_data_dictionary.csv
    │
    ├── 📁 ETL_Scripts/
    │   ├── etl_railway_pro.py
    │   └── generate_inference.py
    │
    ├── 📁 DWH/
    │   ├── DWH.png
    │   └── Staging table.png
    │
    ├── 📁 ML_Models/
    │   ├── railway_delay_model_final_balanced.rar
    │   └── generate_predicted_data.py
    │   └── synthetic_future_predictions.csv 
    │   └── ML_model.py
    │
    ├── 📁 PowerBI/
    │   └── railway power bi.pbix
    │   └── railway power bi.pdf
    │
    └── README.md
    └── project architecture.jpeg

------------------------------------------------------------------------

## System Architecture & Workflow

The design follows a **Hybrid Data Warehouse** integrating **historical facts** with **predictive insights**.

### 1️⃣ Data Ingestion & ETL (Python)
Scripts: `ETL_Scripts/etl_railway_pro.py`, `ETL_Scripts/generate_inference.py`

- Cleans raw CSV data  
- Handles missing values and standardizes station names  
- Creates engineered features (e.g., actual vs scheduled duration)  
- Outputs cleaned data for warehouse ingestion  

---

### 2️⃣ Machine Learning Modeling (Python / Scikit-Learn)
- Model: **Random Forest / ML_model.py**  
- Input: Journey Date, Departure Time, Station Route, Ticket Class, Railcard Type  
- Target: Binary Classification (Delayed / On Time)  
- Generates: `ML_Models/synthetic_future_predictions.csv`  
- Model archive: `railway_delay_model_final_balanced.rar`

---

### 3️⃣ Data Warehousing (DWH)
- Folders: `DWH/`  
- Contains visuals: `DWH.png` and `Staging table.png`  
- Data modeled in **Star Schema**  
- Stores historical and predictive insights  

---

### 4️⃣ Visualization (Power BI)
- Folder: `PowerBI/`  
- Dashboard file: `railway power bi.pbix`  
- PDF export: `railway power bi.pdf`  
- Displays historical trends, delay hotspots, and future risk predictions  

------------------------------------------------------------------------

## How to Run the Project

### **Step 1 --- Generate ML Predictions**

``` bash
python ETL_Scripts/generate_inference.py
```

### **Step 2 --- Build Data Warehouse**

1.  Run schema creation script\
2.  Load cleaned CSV + prediction CSV into staging\
3.  Execute fact load script

### **Step 3 --- Power BI Dashboard**

-   Open `Railway_Dashboard.pbix`
-   Refresh SQL connection\
-   Explore "Future Risk" page

------------------------------------------------------------------------

## Technologies Used

-   **Python:** Pandas, NumPy, Scikit-learn\
-   **SQL Server:** T-SQL, Data Warehousing\
-   **Power BI:** DAX, Data Modeling\
-   **Concepts:** ETL, Star Schema, Feature Engineering, ML
    Classification

------------------------------------------------------------------------

## Contact

For questions or improvements, feel free to connect!
