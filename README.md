# README

## Digital Watermarking using Mobius Transform in DCT Domain

This project demonstrates a novel approach for embedding and extracting digital watermarks in Discrete Cosine Transform (DCT) coefficients using the Mobius Transform. The technique aims to provide robust watermarking that can resist common image processing attacks while maintaining high fidelity in watermark extraction.

### Requirements

To run the project, you need the following Python libraries:
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- scikit-image (`skimage`)
- scikit-learn (`sklearn`)
- DEAP (`deap`)

You can install these dependencies using pip:
```bash
pip install opencv-python-headless numpy matplotlib scikit-image scikit-learn deap
```

### Usage

The project includes functions for embedding and extracting watermarks using the Mobius Transform applied to DCT coefficients, optimizing the Mobius Transform parameters using a Genetic Algorithm (GA), and evaluating the performance of the watermarking technique.

#### Functions Overview

1. **Mobius Transform Functions**
   - `mobius_transform(x, y, a, b, c, d)`: Applies the Mobius Transform to coordinates `(x, y)` with parameters `a, b, c, d`.
   - `inverse_mobius_transform(new_x, new_y, a, b, c, d)`: Applies the inverse Mobius Transform to coordinates `(new_x, new_y)` with parameters `a, b, c, d`.

2. **Watermark Embedding and Extraction**
   - `embed_watermark_dct_mobius(original_image, watermark, alpha, mobius_params)`: Embeds the watermark into the DCT coefficients of the original image using the Mobius Transform.
   - `extract_watermark_dct_mobius(watermarked_image, original_image, alpha, mobius_params)`: Extracts the watermark from the DCT coefficients of the watermarked image using the Mobius Transform.

3. **Performance Metrics**
   - `calculate_ber(original_watermark, extracted_watermark)`: Calculates the Bit Error Rate (BER) between the original and extracted watermarks.
   - `calculate_ncc(original_watermark, extracted_watermark)`: Calculates the Normalized Cross-Correlation (NCC) between the original and extracted watermarks.

4. **Optimization**
   - `optimize_mobius_parameters_DEAP(original_image, watermark_image)`: Optimizes the Mobius Transform parameters using a Genetic Algorithm to maximize the Structural Similarity Index Measure (SSIM) between the original and extracted watermarks.

5. **Image Processing**
   - `process_image_mobius(input_image_path, watermark_image_path, mobius_params, fake_disappear=False)`: Embeds and extracts a watermark from an image using the optimized Mobius Transform parameters and evaluates the performance.

#### Running the Code

To use the watermarking technique on an image, follow these steps:

1. **Load the Images**
   - Load the original image and watermark image using OpenCV.

2. **Optimize Mobius Transform Parameters**
   - Use `optimize_mobius_parameters_DEAP` to find the best parameters for embedding the watermark.

3. **Embed and Extract Watermark**
   - Use `process_image_mobius` to embed the watermark into the original image and extract it.

4. **Evaluate Performance**
   - The script evaluates the performance using SSIM, PSNR, MSE, BER, and NCC metrics.

Example code to run the process:
```python
import cv2
import numpy as np

# Load the original image and watermark image
image_path = 'cameraman.tif'
watermark_path = 'watermark_image.png'

# Resize the images as needed
desired_size_original = (256, 256)
original_image = cv2.resize(cv2.imread(image_path), desired_size_original)

desired_size_watermark_1 = (32, 32)
watermark_image_1 = cv2.resize(cv2.imread(watermark_path), desired_size_watermark_1)

desired_size_watermark_2 = (256, 256)
watermark_image_2 = cv2.resize(cv2.imread(watermark_path), desired_size_watermark_2)

# Optimize Mobius Transform parameters
optimized_params_1 = optimize_mobius_parameters_DEAP(original_image, watermark_image_1)

# Embed and extract watermark
if optimized_params_1 is not None:
    extracted_watermark_path_1 = process_image_mobius(image_path, watermark_path, optimized_params_1, fake_disappear=True)
    print(f"Extracted Watermark (Size 1) saved at: {extracted_watermark_path_1}")
else:
    print("Optimization for Watermark (Size 1) did not converge to a solution.")

optimized_params_2 = optimize_mobius_parameters_DEAP(original_image, watermark_image_2)

if optimized_params_2 is not None:
    extracted_watermark_path_2 = process_image_mobius(image_path, watermark_path, optimized_params_2, fake_disappear=True)
    print(f"Extracted Watermark (Size 2) saved at: {extracted_watermark_path_2}")
else:
    print("Optimization for Watermark (Size 2) did not converge to a solution.")
```

### Conclusion

This project demonstrates the application of the Mobius Transform in the DCT domain for robust digital watermarking. The use of Genetic Algorithms for optimizing the transform parameters enhances the performance of the watermarking technique, making it resilient to common image processing attacks while maintaining high image quality. This approach is promising for digital image authentication and copyright protection.
