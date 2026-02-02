# SpaceX-falcon-9-first-stage-landing-prediction
Prediction of Space X Falcon 9 First Stage Landing

Space X advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars; other providers cost upward of 165 million dollars each, much of the savings is because Space X can reuse the first stage. Therefore if we can determine if the first stage will land, we can determine the cost of a launch. This information can be used if an alternate company wants to bid against space X for a rocket launch. *Coursera*

## Project Structure

-   `scripts/dash_app.py`: Main dashboard application.
-   `scripts/build_model.py`: Script to train and save the ML model.
-   `data/`: Contains dataset CSVs and the serialized model (`model.pkl`).
-   `notebooks/`: Jupyter notebooks for data collection, wrangling, and EDA.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running Locally

1.  Navigate to the `scripts` directory (or root):
    ```bash
    python scripts/dash_app.py
    ```
2.  Open your browser and visit `http://127.0.0.1:8050/`.

<!-- ## Running with Docker

1.  Build the image:
    ```bash
    docker build -t spacex-dash-app .
    ```
2.  Run the container:
    ```bash
    docker run -p 8050:8050 spacex-dash-app
    ```

## Deployment

The project includes a `Procfile` for deployment on platforms like Heroku. -->
