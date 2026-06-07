# Pathogen Suppression by Bacterial Communities

This project, associated with the *Ratzke Lab*, investigates how bacterial communities of varying sizes and nutrient conditions collectively suppress a pathogen of interest. 

---

## 🔬 Dataset Generation
I generated synthetic datasets based on the microbial consumer resource model as described in this [Scientific Reports publication](https://doi.org/10.1038/s41598-020-60130-2). 

> *Note: Multiple iterations were performed to optimize the dataset features, which can be tracked across the repository history.*

## 🤖 Machine Learning Objective
The core goal is to determine if it is possible to predict community pathogen suppression in large synthetic communities using *ONLY* the varying nutrient features as inputs. 

I evaluated and compared *6 tabular machine learning models*:
1. *Linear Baseline:* Ridge Regression
2. *Tree-Based Ensembles:*
   * XGBoost
   * RandomForestRegressor
   * LightGBM
   * CatBoost
3. *TabPFN-3:* *(the latest model by [Prior Labs](https://priorlabs.ai/))* 

## 🎯 Project Impact
This entire exercise serves as a *logical baseline*. By establishing a clear benchmark on synthetic data, we can validate these models before testing them on real-world wet lab data to see if our conclusions hold true in vivo!



