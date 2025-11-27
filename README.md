# End-to-End Railway Delay Prediction & Data Warehousing Project

![System Architecture](project_architecture.jpeg)

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
    │   └── synthetic_future_predictions.csv  
    │
    ├── 📁 ETL_Scripts/
    │   ├── etl_railway_pro.py
    │   └── generate_inference.py
    │
    ├── 📁 SQL_Warehouse/
    │   ├── 01_DWH_Schema.sql
    │   └── 02_Load_Facts.sql
    │
    ├── 📁 ML_Models/
    │   ├── railway_delay_model.pkl
    │   └── training_notebook.ipynb
    │
    ├── 📁 PowerBI/
    │   └── Railway_Dashboard.pbix
    │
    └── README.md

------------------------------------------------------------------------

## System Architecture & Workflow

The design follows a **Hybrid Data Warehouse** integrating **historical
facts** with **predictive insights**.

### 1️⃣ Data Ingestion & ETL (Python)

Script: `etl_railway_pro.py`

-   Cleans raw CSV data\
-   Fixes nulls and inconsistent station names\
-   Creates engineered features\
-   Outputs clean data for SQL ingestion

------------------------------------------------------------------------

### 2️⃣ Machine Learning Modeling (Scikit-Learn)

-   Model: **Random Forest Classifier**\
-   Target: **Delay (Yes/No)**\
-   Outputs: `synthetic_future_predictions.csv`\
-   Predictions include:
    -   Binary output\
    -   Probability score

------------------------------------------------------------------------

### 3️⃣ Data Warehousing (SQL Server)

**Modeling: Multi-Fact Star Schema**

Created Tables: - `Dim_Date` - `Dim_Station` - `Dim_Ticket_Details` -
`Fact_Railway_Ticket_Sales`\
- `Fact_Future_Risk`

Scripts:\
- `01_DWH_Schema.sql` → Create schema\
- `02_Load_Facts.sql` → Load predictions

------------------------------------------------------------------------

### 4️⃣ Visualization (Power BI)

**Dashboard Features:** - Historical performance trends\
- Delay rate heatmaps\
- Future risk predictions\
- Route-level risk ranking

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
