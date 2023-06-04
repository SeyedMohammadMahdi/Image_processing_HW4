![](Aspose.Words.0d4c4223-d24f-49f8-bd3c-4dd1bbacbb45.001.png)

EXERCISES 

Instructions  Appendix: Code with comments.

Delivery  • All homework should be sent through VU (No Telegram, Email, etc.). 

- The report should be in PDF format named “Number of Homework-First Name Last Name.pdf”. 
- Notice the deadlines. 

Points  • Don't use MATLAB, Python, etc. library/toolbox for solving problems (Except 

math functions) 

- Utilize functions when explicitly mentioned in the question. 
- Any form of plagiarism will not be entertained and will result in a loss of grade. 
- You can compare your own result by MATLAB, etc. output (optional).
1. Color 
1. Color space 
1. Convert Lena to HSV format, and display the H, S, V components as separate grayscale images.  
1. Present Lena in a new color space which was not introduced in class. Then convert R, G and B to the new color space components manually. 
2. Quantization 

1\.2.1. We want to weave the Baboon image on a rug. To do so, we need to reduce the number of colors in the image with minimal visual quality loss. If we can have 32, 16 and 8 different colors in the weaving process, reduce the color of the image to these three special modes. Compare using MSE and PSNR and display the results.

2. Features 
1. Harris Corner Detector 
   1. Extract interest points using the Harris Corner detector that you implemented. In this way, apply the Harris Corner detector for at least 4 different scales. Which interest points do you observe to be detected across all these different scales? Notice that your implementation should allow for any suitable scale as input, however you can show results on a minimum of 4 different scales (Test on harris.JPG Image). 
2. Scene stitching with SIFT/SURF features 

2\.2.1. Use the OpenCV implementation of the SIFT or SURF operator to find interest points and establish correspondences between the images. In this case you can directly compare the feature vectors of interest points. You will match and align between different views of a scene with SIFT/SURF features. Discuss results and demonstrates the output of each method separately (Test on sl, sm and sr.jpg images). 
