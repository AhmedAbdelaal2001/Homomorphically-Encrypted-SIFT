# A Secure Image Processing Pipeline based on Homomorphic Encryption

## Overview
This repository contains the implementation of a privacy-preserving image processing system using homomorphic encryption, specifically the Paillier Cryptosystem. Our project enables operations on encrypted images, eliminating the need for decryption during processing. This approach ensures the privacy and security of image data. An example of the power of this idea is shown below.

![intro1](https://github.com/AhmedAbdelaal2001/Homomorphically-Encrypted-Image-Processing/assets/101427765/174ceff0-922f-4c7e-a265-4d3e09c00d00)

This figure shows 2 encrypted images being matched directly in the encrypted domain, where any external observer cannot tell the original contents of the image. The original images contain 2 shots of the Pyramids of Giza, justificating the traingular structure detected by the keypoints. How this operation was performed will be discussed in the upcoming sections.


## Features
- **Paillier Cryptosystem**: Implementation of encryption, decryption, and key generation based on the Paillier Cryptosystem.
- **Homomorphic Operations**: Support for addition and scalar multiplication on ciphertexts.
- **Image Processing Operations**: Implements various image processing tasks like smoothing, edge detection, and morphology on encrypted images.
- **HESIFT Algorithm**: Adaptation of the Scale Invariant Feature Transform (SIFT) algorithm for feature extraction from encrypted images.
- **Image Matching**: Utilizes the Fast Library for Approximate Nearest Neighbors (FLANN) for efficient image matching.

## HESIFT Showcase:
### Encrypted DoG Space:
![DoG_encrypted](https://github.com/AhmedAbdelaal2001/Homomorphically-Encrypted-Image-Processing/assets/101427765/93a67391-9dcb-4645-bc47-4d715174c248)

### Decrypted DoG Space: 
![DoG](https://github.com/AhmedAbdelaal2001/Homomorphically-Encrypted-Image-Processing/assets/101427765/2ea8e5c7-00af-41c1-b59a-b3c3c689094f)

### Keypoints: 
![birdKeypoints](https://github.com/AhmedAbdelaal2001/Homomorphically-Encrypted-Image-Processing/assets/101427765/37620937-951f-4ae6-b946-225402f6676a) ![pyramidsKeypoints](https://github.com/AhmedAbdelaal2001/Homomorphically-Encrypted-Image-Processing/assets/101427765/605876dc-2056-4081-b57b-304b387e38cc)



## Code Structure
The structure of the code is as follows:
- The "utils_encryptedDomain" folder contains our implementation of the Paillier Cryptosystem, specifically tuned for efficiency and compatability with NumPy Arrays. Furthermore, it contains our implementation of Convolutions in the encrypted domain, as well as any other operations that can potentially be performed on encrypted images
- The "feature_extractors" folder contains our implementation of Lowe's SIFT algorithm on encrypted images, named HESIFT (Homomorphically Encrypted SIFT). The HESIFT function found at the end of the file can be directly called to extract the keypoints and descriptors.

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

### License

This project is licensed under the [MIT License](LICENSE).


