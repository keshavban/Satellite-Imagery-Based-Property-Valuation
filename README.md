# Satellite-Imagery-Based-Property-Valuation

## OVERVIEW

A Real Estate Analytics firm aims to improve its valuation framework by developing a **Multimodal Regression Pipeline** that predicts property market value using both tabular data and satellite imagery.

You are provided with historical housing data (including coordinates) and must programmatically acquire visual data to capture environmental context. The goal is to build a model that accurately values assets by integrating "curb appeal" and neighborhood characteristics (like green cover or road density) into traditional pricing models.

This project moves beyond standard data analysis by challenging you to combine two different types of data—numbers and images—into a single, powerful predictive system.

## OBJECTIVE

- ***Build a multimodal regression model*** to predict property value (Target: Price).
- ***Programmatically acquire satellite imagery*** using latitude/longitude coordinates to capture visual environmental context.
- ***Perform exploratory and geospatial analysis*** to understand how visual factors (e.g., proximity to water, density) influence price.
- ***Engineer features*** using Convolutional Neural Networks (CNNs) to extract high-dimensional visual embeddings from the images.
- ***Test and compare fusion architectures*** (e.g., combining image data and tabular data at different stages) to find the most accurate method.
- ***Ensure Model Explainability*** by using tools like Grad-CAM to visually highlight the specific areas in the satellite imagery that influenced the model's price prediction.
