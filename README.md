# Quick Start

Follow these steps to run the application:

1. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Fetch/Update Data (If Necessary):**  
   Ensure the latest CSV data files (e.g., `ahmedabad.csv`) are in the `app/data/` directory. If needed, run the script to fetch data:
    ```bash
    python "app\data\data.py"
    ```

3. **Train Models (Required):**  
   Before starting the server, train the models. This creates the necessary `.pkl` files in `app/models/`.
    ```bash
    python -m app.ml.train_models
    ```
    *(This step can take some time.)*

4. **Start Backend Server:**
    ```bash
    uvicorn app.main:app --reload
    ```
    Keep this terminal open. The API runs at `http://127.0.0.1:8000`.

5. **Open Frontend:**  
   Open the `frontend/index.html` file in your web browser.

---

## API Endpoint
- **GET `/predict`**
    - **Query Parameters:**
        - `city` (str, required): e.g., `ahmedabad`
        - `model_name` (str, required): e.g., `LightGBM`, `Ensemble`
        - `forecast_type` (str, required): `48h`, `1week`, `2weeks`
        - `day_of_week` (int, optional): 0-6 (Mon-Sun), only for `1week` or `2weeks` type.
    - **Returns:** JSON array of forecast objects.
