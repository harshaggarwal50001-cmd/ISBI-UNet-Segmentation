# ISBI2012 U-Net Image Segmentation

This project uses a U-Net based deep learning model for image segmentation on the ISBI2012 dataset.

The project has a Streamlit frontend, FastAPI backend and PostgreSQL database.

## Project Structure

ISBI2012-UNet/

 backend/                 # FastAPI backend
 frontend/                # Streamlit frontend                  
 model_code.py            # Model related code
 requirements.txt         # Python dependencies
.gitignore
 README.md

The trained model is not stored directly in the Git repository. It is available in the GitHub Releases section.

## Technologies Used

* Python
* PyTorch
* FastAPI
* Streamlit
* PostgreSQL
* Docker
* SQLAlchemy

## Model

The pretrained U-Net model is provided as a release asset.

Download best_unet_model.pth from the **Releases** section of this repository.

After downloading it, place it in the project root:

ISBI2012-UNet/
 backend/
 frontend/
 model_code.py
 unet_model.pth
 requirements.txt

## Setup

Clone the repository:
git clone  https://github.com/harshaggarwal50001-cmd/ISBI-UNet-Segmentation.git;
cd ISBI-UNet-Segmentation

Create a virtual environment:
python -m venv venv

Activate it:

venv\Scripts\activate

Install the required packages:
pip install -r requirements.txt

## Environment Variables

The database connection URL is stored in a *.env* file.

Create a .env file in the project root and add:

DATABASE_URL=your_database_url

## PostgreSQL

The project uses PostgreSQL for storing prediction related information.

PostgreSQL is run using Docker. Make sure Docker Desktop is installed and the PostgreSQL container is running before starting the backend.

## Running the Backend

Start the FastAPI backend using:

uvicorn backend.main:app --reload

The API will be available at:

http://127.0.0.1:8000


## Running the Frontend

Start the Streamlit application using the frontend file:

streamlit run frontend/app.py

## Project Workflow

The basic workflow of the project is:
Input Image -> Streamlit Frontend -> FastAPI Backend -> U-Net Model -> Segmentation Prediction -> PostgreSQL


## Evaluation

Model's segmentation results are evaluated using Dice Score, Intersection over Union (IoU), and Rand Score.

## Notes

* The trained model file is available in GitHub Releases.
* The *.env* file is not uploaded to GitHub.
* The *venv* folder is also not uploaded.
