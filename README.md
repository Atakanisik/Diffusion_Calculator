# Diffusion_Calculator
Comprehensive IVIM calculator tool

[![DOI](https://zenodo.org/badge/1116794592.svg)](https://doi.org/10.5281/zenodo.17939646)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/Made%20with-MATLAB-orange)](https://www.mathworks.com/products/matlab.html)

**Diffusion Calculator** is a comprehensive, open-source software suite designed for the quantitative analysis of renal **Intravoxel Incoherent Motion (IVIM)** MRI data.

This app introduces **AI-powered segmentation (MedSAM)**, a standalone executable for easy installation, and rigorous clinical validation against industry standards (Siemens Syngo.via).

![Splash Interface](assets/splash_icon.png)

---

##  Key Features

* **Verified Accuracy:** Benchmarked against FDA-approved **Siemens Syngo.via**. Demonstrated **>0.90 Dice Similarity Coefficient (DSC)**.
* **AI-Assisted Segmentation:** Integrates **MedSAM** to automate kidney localization, replacing tedious manual ROI drawing.
* **Artifact Correction:** Features a novel **"Safe-Zone Adaptive Thresholding"** algorithm to eliminate cortex/medulla edge artifacts.
* **Universal Compatibility:** Supports DICOM datasets from major vendors (Siemens, GE, Philips) with automated b-value parsing.
* **Advanced Fitting Models:** Mono-exponential (ADC), Bi-exponential (IVIM), Segmented, Bayesian, and Tri-exponential.
* **Universal Export:** Exports parameter maps as **NIfTI (.nii)** and **Excel (.xlsx)**.

---

##  Downloads & Installation

We provide a standalone version for clinicians and source code for developers.

### 1. Standalone Executable (No MATLAB Required)
*Recommended for clinical users.*
The standalone package includes the compiled application and necessary runtime installers.

* **Download Installer:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17941316.svg)](https://doi.org/10.5281/zenodo.17941316)

**Installation Steps:**
1.  Download and unzip the installer package.
2.  Run `Installer.exe` or `Web_Installer.exe`.
3.  Follow the prompts (it will automatically download MATLAB Runtime R2023a if you run `Web_Installer.exe` ).
4.  Launch the application from your desktop.

### 2. Example Dataset
To test the software, you can download our anonymized validation dataset (DICOM):

* **Download Data:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17941231.svg)](https://doi.org/10.5281/zenodo.17941231)

### 3. Source Code (For Developers)
*Requires MATLAB R2023a + Toolboxes.*
* Clone this repository or download the source archive

---

## 📺 Video Tutorial

For a comprehensive walkthrough of the workflow (Loading, Registration, Segmentation, and Analysis), please download and watch our tutorials:

* **Watch Tutorial:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17943598.svg)](https://doi.org/10.5281/zenodo.17943598)

---


## 🧮 Mathematical Models

The software implements non-linear least squares fitting (Levenberg-Marquardt) for the following models:

### 1. Bi-Exponential IVIM Model
$$\frac{S(b)}{S_0} = f \cdot e^{-b D^*} + (1-f) \cdot e^{-b D}$$

### 2. Mono-Exponential (ADC)
$$S(b) = S_0 \cdot e^{-b \cdot ADC}$$

### 3. Tri-Exponential Model
$$\frac{S(b)}{S_0} = f_{fast} e^{-b D_{fast}} + f_{inter} e^{-b D_{inter}} + f_{slow} e^{-b D_{slow}}$$


---



**License:** MIT License.  
**Contact:** [atakani@baskent.edu.tr]
