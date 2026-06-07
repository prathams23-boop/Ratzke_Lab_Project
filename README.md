{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "83547bdb-69d7-47e5-9470-2a6b2513f47a",
   "metadata": {},
   "source": [
    "# Pathogen Suppression by Bacterial Communities\n",
    "\n",
    "This project, associated with the **Ratzke Lab**, investigates how bacterial communities of varying sizes and nutrient conditions collectively suppress a pathogen of interest. \n",
    "\n",
    "---\n",
    "\n",
    "## 🔬 Dataset Generation\n",
    "I generated synthetic datasets based on the microbial consumer resource model as described in this [Scientific Reports publication](https://doi.org/10.1038/s41598-020-60130-2). \n",
    "> *Note: Multiple iterations were performed to optimize the dataset features, which can be tracked across the repository history.*\n",
    "\n",
    "## 🤖 Machine Learning Objective\n",
    "The core goal is to determine if it is possible to predict community pathogen suppression in large synthetic communities using **ONLY** the varying nutrient features as inputs. \n",
    "\n",
    "I evaluated and compared **6 tabular machine learning models**:\n",
    "1. **Linear Baseline:** Ridge Regression\n",
    "2. **Tree-Based Ensembles:** * XGBoost\n",
    "   * RandomForestRegressor\n",
    "   * LightGBM\n",
    "   * CatBoost\n",
    "3. **TabPFN-3:** *(the latest model by [Prior Labs](https://priorlabs.ai/))*\n",
    "\n",
    "---\n",
    "\n",
    "## 🎯 Project Impact\n",
    "This entire exercise serves as a **logical baseline**. By establishing a clear benchmark on synthetic data, we can validate these models before testing them on real-world wet lab data to see if our conclusions hold true in vivo!"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
