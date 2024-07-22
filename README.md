# Image Colorizer

This repository contains a FastAPI-based web application for colorizing black-and-white images using pre-trained GAN models. The application allows users to upload a black-and-white image and select a model to generate the colored version of the image.        

Installation
------------

1.  Clone the repository:
    
        git clone https://github.com/eshiofune/colorization.git
        cd colorization/app
    
2.  Create a virtual environment and activate it:
    
        python -m venv env
        source env/bin/activate  # On Windows use `env\Scripts\activate`
    
3.  Install the required packages:
    
        pip install -r requirements.txt
    

Usage
-----

1.  Start the FastAPI server:
    
        uvicorn main:app --reload
    
2.  Open your web browser and go to `http://localhost:8000` to access the web application.


Example
-------

#### Black and White Image
![Black and White Image](examples/landscape_bnw.jpeg)

#### Colored Image
![Colored Image](examples/landscape_color.png)