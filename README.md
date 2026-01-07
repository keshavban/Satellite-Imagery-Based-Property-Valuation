**Satellite Imagery--Based Property Valuation**
==============================================

This project explores **property price prediction using a combination of satellite imagery and structured tabular data**. The goal is to assess whether visual information from satellite images can complement traditional tabular features (such as size, location, and condition) and improve predictive performance beyond tabular-only baselines.

The project follows a **progressive experimental pipeline**, beginning with data understanding and baseline models, and culminating in a **final multimodal deep learning architecture using EfficientNet for image feature extraction**.

* * * * *

**Project Objectives**
----------------------

1.  Understand the financial and spatial drivers of property prices using tabular data.

2.  Evaluate whether satellite imagery alone contains meaningful price-related signals.

3.  Design and train a multimodal model that fuses image embeddings with tabular features.

4.  Compare multimodal performance against tabular-only baselines.

5.  Interpret model behavior using Grad-CAM visual explanations.

6.  Select a final model for inference on unseen test data.

* * * * *

**Final Project Structure**
---------------------------

`Satellite-Imager-Based-Property-Valuation/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── images/
│
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── image_sanity_check.ipynb
│   ├── image_only_model.ipynb
│   ├── Tabular_only_models.ipynb
│   ├── initial_multimodal.ipynb
│   └── final_multimodal.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py
│   ├── datasets.py
│   └── models.py
│
├── output/
│
├── .gitignore
└── README.md`

* * * * *

**Directory & File Description**
--------------------------------

* * * * *

**1\. `data/`**
---------------

### `data/raw/`

Contains the original dataset provided for the project, including:

-   Property attributes

-   Target variable (log-transformed price)

These files are preserved in their original form to ensure reproducibility.

### `data/processed/`

Contains processed datasets generated after:

-   Feature selection

-   Feature engineering (e.g., house age, ratios)

-   Normalization / standardization

All models consume data from this directory rather than modifying raw files directly.

* * * * *

**2\. `images/`**
-----------------

Contains satellite images corresponding to each property.

-   Image filenames are indexed to align with the tabular dataset.

-   A small number of missing images were detected during sanity checks and handled safely during dataset construction.

* * * * *

**3\. `notebooks/`**
--------------------

This directory contains the **full experimental workflow**, executed in logical stages.

* * * * *

### **`data_understanding.ipynb`**

This notebook focuses on **exploratory data analysis (EDA)** and financial intuition.

Key analyses include:

-   Distribution of house prices and log-price transformation

-   Relationship between living area and price

-   Impact of location (latitude/longitude) on valuation

-   Effect of property condition, grade, and amenities

-   Correlation analysis between engineered features and price

The EDA confirms that **tabular features already encode strong economic signals**, motivating their role as a core input to all subsequent models.

* * * * *

### **`image_sanity_check.ipynb`**

This notebook verifies:

-   Correct alignment between tabular rows and image files

-   Detection of missing or corrupt images

-   Visual inspection of random satellite samples

This step ensures data integrity before training any image-based models.

* * * * *

### **`Tabular_only_models.ipynb`**

This notebook establishes **tabular-only baselines**.

Models include:

-   Simple regression baselines

-   Tree-based models (e.g., XGBoost)

Results:

-   Tabular-only models achieve **predictive performance (R² ≈ 0.89)**.

-   Confirms that structured data is highly informative for property valuation.

These models serve as the **primary benchmark** for evaluating multimodal approaches.

* * * * *

### **`image_only_model.ipynb`**

This notebook evaluates whether satellite imagery alone can predict property prices.

Approach:

-   CNN-based regression model trained only on images

-   No tabular inputs provided

Results:

-   R² ≈ 0

-   Indicates that satellite imagery alone lacks sufficient signal for accurate price prediction

This experiment motivates **multimodal fusion**, rather than image-only modeling.

* * * * *

### **`initial_multimodal.ipynb`**

This notebook represents the **first multimodal attempt**, combining:

-   CNN-based image embeddings

-   Tabular feature embeddings via an MLP

-   Feature fusion followed by a regression head

Key characteristics:

-   CNN backbone kept frozen

-   Full tabular feature set used

Results:

-   Stable training

-   R² ≈ **0.77**

-   Demonstrates that combining modalities improves over image-only models but does not yet match strong tabular baselines

This model serves as a **stepping stone** toward the final architecture.

* * * * *

### **`final_multimodal.ipynb`**

This notebook contains the **final and best-performing multimodal model**.

#### Architecture:

-   **EfficientNet** pretrained backbone for image feature extraction

-   CNN kept frozen to avoid overfitting

-   MLP-based tabular encoder

-   Concatenation-based fusion layer

-   Fully connected regression head

#### Training Strategy:

-   Controlled learning rate

-   Regularization via dropout

-   Careful normalization of tabular features

#### Results:

-   Achieves **R² ≈ 0.912**

-   Performance overcomes strong tabular-only baselines

-   Demonstrates meaningful contribution of image context when combined correctly

This model is selected as the **final model for test-set prediction**.

* * * * *

**4\. `src/`**
--------------

Reusable Python modules used across notebooks.

* * * * *

### **`data_fetcher.py`**

Handles:

-   Loading CSV files

-   Managing data paths

-   Centralized access to datasets

* * * * *

### **`datasets.py`**

Defines PyTorch dataset classes:

-   Joint loading of image and tabular data

-   Safe handling of missing images

-   Application of transforms

-   Returns `(image, tabular, target)` tuples

* * * * *

### **`models.py`**

Defines all neural architectures:

-   Tabular MLP

-   CNN image encoders

-   Multimodal fusion networks

-   Final EfficientNet-based model

* * * * *

**5\. `output/`**
-----------------

Stores:

-   Model predictions

-   Visualizations

-   Grad-CAM overlays

-   Any generated result artifacts

* * * * *

**Model Performance Summary**
-----------------------------

| Model | R² |
| --- | --- |
| Final Multimodal (EfficientNet + Tabular) | **~0.91** |
| Tabular-only (baseline) | **~0.89** |
| Initial Multimodal (Frozen CNN) | ~0.77 |
| Image-only CNN | ~0.00 |

* * * * *

**Explainability: Grad-CAM**
----------------------------

Grad-CAM is applied **only to the multimodal model**, focusing on the image branch.

Observations:

-   The model attends to roads, building clusters, and neighborhood layout

-   Confirms that visual context contributes to final predictions

-   Supports interpretability of the multimodal approach

* * * * *

**Final Model Selection**
-------------------------

The **final multimodal EfficientNet-based model** is chosen for inference on the test dataset because:

-   It integrates visual and tabular information effectively

-   It generalizes well without overfitting

-   It aligns with the project's core objective of multimodal learning

-   It provides visual interpretability through Grad-CAM

* * * * *

**Conclusion**
--------------
In the initial phase of this study, the tabular-only baseline model demonstrated relatively stronger performance, suggesting that structured numerical features were effective in capturing coarse property-level information. However, after incorporating satellite imagery into the learning pipeline, the multimodal model significantly outperformed the tabular-only approach across all key evaluation metrics. This improvement highlights the critical role of visual spatial context---such as surrounding infrastructure, land-use patterns, road connectivity, and neighborhood density---which cannot be adequately represented through tabular data alone. By fusing deep visual features extracted via a convolutional neural network with structured tabular representations through a unified regression head, the model was able to learn complementary and non-redundant information, leading to superior generalization and reduced prediction error. These results validate the effectiveness of multimodal learning for real-estate price estimation and demonstrate that satellite imagery provides substantial predictive value beyond traditional structured features.