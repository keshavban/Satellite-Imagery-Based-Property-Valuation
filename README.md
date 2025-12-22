# Satellite Imagery–Based Property Valuation

## Overview
This project develops a **multimodal regression system** to predict residential property prices by combining traditional tabular housing data with **satellite imagery**. The motivation is to enhance property valuation models by incorporating visual and environmental context—such as neighborhood layout, road connectivity, and green cover—into standard numerical pricing frameworks.

The project goes beyond conventional regression by integrating **computer vision (CNNs)** with **structured data models**, and by providing **model explainability** using Grad-CAM.

---

## Dataset
### Tabular Data
- Source: Kaggle House Sales Dataset
- Key Features:
  - Bedrooms, bathrooms, floors
  - Sqft living area and lot size
  - Location (latitude, longitude)
  - Condition, grade, waterfront, view
- Target:
  - Property price (log-transformed for modeling)

### Satellite Imagery
- Satellite images were programmatically fetched using **ESRI World Imagery**
- Each image corresponds to a property’s latitude and longitude
- Images capture:
  - Road density
  - Neighborhood structure
  - Green cover and open areas
  - Water proximity

---

## Project Pipeline
1. Exploratory Data Analysis (EDA)
2. Feature Engineering on tabular data
3. Strong tabular baselines (Linear, Random Forest, XGBoost)
4. Satellite image acquisition and preprocessing
5. CNN-based image feature extraction (ResNet18)
6. Multimodal fusion of image and tabular embeddings
7. Model explainability using Grad-CAM
8. Final evaluation and comparison

---

## Results Summary
| Model | RMSE (log) | R² (log) |
|------|-----------|----------|
| Linear Regression | 0.252 | 0.769 |
| Random Forest | 0.179 | 0.883 |
| XGBoost | **0.172** | **0.893** |
| Multimodal (CNN + Tabular) | 0.595 | -0.28 |

While XGBoost achieved the best predictive accuracy, the multimodal model provided **interpretable visual insights** into environmental factors influencing property value.

---

## Explainability
Grad-CAM visualizations show that the CNN focuses on:
- Road networks and accessibility
- Neighborhood boundaries
- Proximity to open areas and water
- Dense residential layouts

These findings align with real-world property valuation intuition.

---

## Tech Stack
- Python
- Pandas, NumPy, GeoPandas
- Scikit-learn, XGBoost
- PyTorch (CNN, multimodal learning)
- OpenCV, Matplotlib, Seaborn

---

## How to Run
1. Install dependencies:
pip install -r requirements.txt
2. Download satellite images:
python src/data_fetcher.py
3. Run notebooks in sequence:
- 01_data_understanding.ipynb
- 02_eda.ipynb
- 03_model_tabular.ipynb
- 04_image_analysis.ipynb
- 05_multimodal_training.ipynb
- 06_gradcam.ipynb

---

## Conclusion
This project demonstrates that while tabular models remain strong for price prediction, satellite imagery provides **valuable contextual and explainable insights**, making the overall valuation system more transparent and informative.
