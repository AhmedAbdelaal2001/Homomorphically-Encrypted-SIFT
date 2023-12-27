# Homomorphically-Encrypted-SIFT

## Image Matching with Paillier Encryption and HE-SIFT

### Introduction

Welcome to the Image Matching project! This project focuses on securely matching images using a combination of Paillier encryption and Homomorphic Encryption-based Scale-Invariant Feature Transform (HE-SIFT). The goal is to provide a safe and privacy-preserving method for image matching while maintaining the integrity of sensitive data.

### How to Run Locally

To run the project locally, follow the steps below:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AhmedAbdelaal2001/Homomorphically-Encrypted-SIFT
   cd your-repo
   ```

2. **Install Required Packages:**
   Ensure that you have the necessary packages installed. You can install them using `pip`:
   ```bash
   pip install django cv2 numpy scikit-image pickle matplotlib
   ```

   Note: Make sure to have Python and pip installed on your system.

3. **Run the Project:**
   ```bash
   python manage.py runserver
   ```

   This command will start the Django development server. Once the server is running, you can access the project at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your web browser.

4. **Explore the Application:**
   - Open your web browser and navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
   - Explore the functionalities related to secure image matching using Paillier encryption and HE-SIFT.

### How to Use

1. **Upload Images:**
   - On the web application, navigate to the image upload section.
   - Upload the image you want to match from your local machine.

2. **Get Results:**
   - After uploading the image, click on the "Match" button.
   - The application will securely process the image using Paillier encryption and HE-SIFT.
   - The result will display the top 5 matches found for the uploaded image.

3. **Review Results:**
   - Examine the matched images and their corresponding details.
   - The results will include information such as similarity scores or other relevant metrics.

### Dependencies

- [Django](https://www.djangoproject.com/): A high-level Python web framework that encourages rapid development and clean, pragmatic design.
- [OpenCV (cv2)](https://opencv.org/): An open-source computer vision and machine learning software library.
- [NumPy](https://numpy.org/): A powerful library for numerical operations in Python.
- [scikit-image](https://scikit-image.org/): A collection of algorithms for image processing.
- [Matplotlib](https://matplotlib.org/): A comprehensive library for creating static, animated, and interactive visualizations in Python.
- [Pickle](https://docs.python.org/3/library/pickle.html): A module for serializing and deserializing Python objects.

### Contributors

- Your Name <your.email@example.com>

### License

This project is licensed under the [MIT License](LICENSE).


