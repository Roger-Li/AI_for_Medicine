Lecture notes for the [AI For Medicine specialization](https://www.deeplearning.ai/ai-for-medicine/) offered by [deeplearning.ai](https://www.deeplearning.ai/).


- [Course 1: AI for Medical Diagnosis](#course-1-ai-for-medical-diagnosis)
  - [1.1. Disease detection with computer vision](#11-disease-detection-with-computer-vision)
    - [1.1.1. Applications of computer vision to medical diagnosis](#111-applications-of-computer-vision-to-medical-diagnosis)
    - [1.1.2. How to handle class imbalance and small training sets](#112-how-to-handle-class-imbalance-and-small-training-sets)
    - [1.1.3. Check how well your model performs](#113-check-how-well-your-model-performs)
  - [1.2. Evaluating models](#12-evaluating-models)
    - [1.2.1. Key evaluation metrics](#121-key-evaluation-metrics)
    - [1.2.2. How does varying the threshold affect evaluation metrics?](#122-how-does-varying-the-threshold-affect-evaluation-metrics)
    - [1.2.3. Interpreting confidence intervals correctly](#123-interpreting-confidence-intervals-correctly)
  - [1.3. Image segmentation on MRI images](#13-image-segmentation-on-mri-images)
    - [1.3.1. MRI data](#131-mri-data)
    - [1.3.2 Image segmentation](#132-image-segmentation)
    - [1.3.3 Practical considerations](#133-practical-considerations)
- [Course 2: AI for Medical Prognosis](#course-2-ai-for-medical-prognosis)
  - [2.1. Linear prognostic models](#21-linear-prognostic-models)
    - [2.1.1. Prognosis and risk](#211-prognosis-and-risk)
  - [2.1.2. Prognostic models in medical practice](#212-prognostic-models-in-medical-practice)
  - [2.4. Representing feature interactions](#24-representing-feature-interactions)
  - [2.5. Evaluating prognostic models](#25-evaluating-prognostic-models)

# Course 1: [AI for Medical Diagnosis](https://www.coursera.org/learn/ai-for-medical-diagnosis)

## 1.1. Disease detection with computer vision

### 1.1.1. Applications of computer vision to medical diagnosis 
- Examples of medical image diagnosis tasks where deep learning algorithms have achieved performance measures comparable to human
  - Dermatology: detecting cancerous skin tissues
  - Ophthalmology: diabetic retinopathy (DR) detection
  - Histopathology: determine cancer spread to Lymph nodes from whole-slide image.

### 1.1.2. How to handle class imbalance and small training sets

- 3 Key challenges
  - Class imbalance
  - Multi-task
  - Dataset size

- Binary cross-entropy loss
$$L(X, y) =  \begin{cases}
    \log P(Y=1|X) \quad \text{if } y = 1\\
    \log P(Y=0|X) \quad \text{if } y = 0
\end{cases}$$

- Weighted loss
  - Let $w_p=\dfrac{\text{num negative}}{\text{num total}}$, and $w_p=\dfrac{\text{num positive}}{\text{num total}}$, and the weighted loss becomes the following

$$L(X, y) =  \begin{cases}
    w_p \times \log P(Y=1|X) \quad \text{if } y = 1\\
    w_n \times \log P(Y=0|X) \quad \text{if } y = 0
\end{cases}$$

- Another way to tackle the class imbalance problem is to use resampling.

- Multi-label / multi-task loss, 
  - e.g., $L(X, y_{\text{mass}}) + L(X, y_{\text{pneumonia}}) + L(X, y_{\text{edema}})$
  - Weighted multi-task loss function
$$L(X, y_{\text{mass}}) =  \begin{cases}
    w_{p, \text{ mass}} \times \log P(Y_{\text{mass}}=1|X) \quad \text{if } y_{\text{mass}} = 1\\
    w_{n, \text{ mass}} \times \log P(Y_{\text{mass}}=0|X) \quad \text{if } y_{\text{mass}} = 0
\end{cases}$$


- Convolutional neural networks (CNN)  architectures
  - Inception-v3
  - ResNet-34
  - [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
  - ResNeXt
  - EfficientNet

- Dataset size problem
  - Use pre-trained CNN and fine-tune deeper layers
  - Generate more samples using data augmentation

### 1.1.3. Check how well your model performs
- Training/validation/test set or training (cross-validation)/test set
- 3 challenges for medical images
  - Patient overlap
    - If multiple data points belong to the same patient, split them into training and test set can lead to over-optimistic test set performance.
    - *Solution*: split data by patient.
  - Set sampling 
    - When sample size is small, we can construct the test set such that at least X% (e.g., 50%) minority class is sampled.
    - Once the test sample is created, we create the validation set next and make it have the same distribution of classes as the test set.
    - Remaining patients in training set.
    - *Solution*: minority class sampling
  - Ground truth / reference standard (in medicine)
    - *Solution*: Consensus voting (in the presence of inter-observer disagreement) or use additional and more definitive medical testing to determine ground-truth.

## 1.2. Evaluating models

### 1.2.1. Key evaluation metrics

- Accuracy
  - Accuracy can be decomposed as follows
  
  $$\text{Accuracy} = P(\text{correct}|\text{disease})\cdot P(\text{disease}) + P(\text{correct}|\text{normal})\cdot P(\text{normal})$$

  - In the presence of class imbalance, accuracy can be dominated by the majority class even though the minority could be what we really care about.

- Sensitivity and Specificity
  - Sensitivity = predict + given disease
  - Specificity = predict - given normal
  - Probability of disease if called *prevalence*
  - Accuracy = Sensitivity $\times$ prevalence + Specificity $\times$ (1 - prevalence)  

- PPV and NPV
  - PPV (positive predictive value) = $P(\text{disease }|+)$
  - NPV (negative predictive value) = $P(\text{normal }|-)$
  - PPV rewritten
    $$PPV = \dfrac{\text{sensitivity}\times\text{prevalence}}{\text{sensitivity}\times\text{prevalence}+(1-\text{specificity})\times(1-\text{prevalence})}$$

- Confusion matrix

  ![Confusion matrix](figures/c1w2_confusion_matrix.png)


### 1.2.2. How does varying the threshold affect evaluation metrics?

- ROC curve
  - Sensitivity versus specificity

### 1.2.3. Interpreting confidence intervals correctly
- Interpretation
  - e.g., with 95% confidence (not 95% probability), $p$ is in the interval [0.72, 0.88]
  - In repeated sampling, the method produces intervals that include the population accuracy in about 95% of samples.
- Use bootstrap to calculate empirical CIs.


## 1.3. Image segmentation on MRI images

### 1.3.1. MRI data 
- MRI example consists of multiple imaging [sequences](https://en.wikipedia.org/wiki/MRI_sequence), which can be combined by treating them as different channels.
- When sequences have misalignment, a preprocessing technique [image registration](https://en.wikipedia.org/wiki/Image_registration) can be applied.


### 1.3.2 Image segmentation
- 2D versus 3D approach
  - 2D approach doesn't consider similarities between adjacent slices (temporal information).
  - 3D approach requires splitting the image slices into blocks / sub-volumes (for computation and memory reason), which preserves temporal information but losses spatial information.

- U-Net
  - [2D U-Net](https://arxiv.org/abs/1505.04597) architecture
  
  ![2D U-Net](figures/c1w3_2d_unet.png)

  - [3D U-Net](https://arxiv.org/abs/1606.06650) architecture, replace 2D operations with 3D counterparts.
  
  ![3D U-Net](figures/c1w3_3d_unet.png)

  - See [reading notes](reading_notes.md##u-net-convolutional-networks-for-biomedical-image-segmentation) on U-Net.

- Data augmentation for segmentation
  - Also need to transform (e.g., rotation, deformation) output segmentation
  - Apply to the 3-D volume
- Loss function for image segmentation
  - Pixel-wise probability estimation
  - *Soft Dice Loss*
  
    $$L(P,G)=1-\frac{2\sum_i^np_ig_i}{\sum_i^np_i^2+\sum_i^ng_i^2}$$ 
    
    where $P$ is the pixel-wise prediction output, $G$ is the ground truth binary labels.
    - Note that the second term is a measure of overlap between $P$ and $G$.

### 1.3.3 Practical considerations
- Different populations and diagnostic technology are challenges for generalization.
- External validation
  - In real-world applications where new population is different from the original population on which the model is developed, we can construct new training/validation/test set and fine-tune the original model.
  - If prospective data is fundamentally different than the retrospective data (e.g., frontal versus lateral chest X-rays), we need to either filter out some of the new data or fine-tune the model.
- Measuring patient outcomes
  - Decision curve analysis
  - Randomized controlled trials
  - Model interpretation



# Course 2: [AI for Medical Prognosis](https://www.coursera.org/learn/ai-for-medical-prognosis)


## 2.1. Linear prognostic models

### 2.1.1. Prognosis and risk

- [Prognosis](https://en.wikipedia.org/wiki/Prognosis) is a medical term that refers to predicting the likelihood or expected development of a disease. 
- Essentially it's a task of predicting risk of a future event.
- Prognosis is useful for
  - Informing patients of their risk of  illness, and survival with illness.
  - Guiding treatment, e.g.,
    - Risk of heart attack $\rightarrow$ Who should get drugs 
    - 6-month mortality risk $\rightarrow$ Who should receive end-of-life care. 

## 2.1.2. Prognostic models in medical practice
- Prognostic model scheme

  ![prog_model_scheme](figures/c2w1_prognostic_model_framework.png)

- Examples
  - Use chads vasc score to predict 1-year risk of stroke for patients with *atrial fibrillation*.
  - Use model for end-stage *liver disease* (MELD) score to estimate 3-month mortality for patients $\geq 12$ of age on liver transplant waiting lists.
  - Use ASCVD Risk Estimator Plus to predict 10-year *risk of heart disease* for patients 20 or older without heart disease.

## 2.4. Representing feature interactions
- Risk equation without interaction, e.g.,
    $$\text{Score} = \ln \text{Age} \times \text{coefficient}_{Age} + \log BP \times \text{coefficient}_{BP}$$
- The same equation with interaction terms
  $$\text{Score} = \ln \text{Age} \times \text{coefficient}_{Age} + \log BP \times \text{coefficient}_{BP} + \ln \text{Age}\times \log BP \times \text{coefficient}_{Age, BP}$$
- The interaction term captures the interdependence of feature effects to the risk, as shown in the figure below (left is without interaction, right is with interaction term)

  ![Interaction term](figures/c2w1_interaction_term.png)


## 2.5. Evaluating prognostic models
- Good risk model should give patient of positive class (e.g., died within 10 years) a higher score than patient of negative class.
- Terminology for pairs
  - For a pair of two patients, if the patient of worse condition has a higher score, then the pair is called **concordant**.
  - If the patient of worse condition has a lower score, the pair is **non-concordant**.
  - If the pair has the same risk scores, it's called **risk ties** regardless of their outcome.
  - If the pair has the same outcome, it's called **ties in outcome** regardless of their risk scores.
  - **Permissible** pairs are pairs with different outcomes.
- Evaluation of prognostic models
  - +1 for a permissible pair that is concordant.
  - +0.5 for a permissible pair for risk time.
- C-Index (AKA *concordance index*)
  - Definition
  $$\text{C-index}=\dfrac{\# \text{ concordant pairs} + 0.5\times\#\text{ risk ties}}{\# \text{ permissible pairs}}$$
  - Interpretation
    - C-index can be interpreted as $P(\text{score}(A) > \text{score}(B)|Y_A > Y_B)$, where random model score = 0.5, and perfect model score = 1.0. 
  - C-index versus AUC
    - Quote the [documentation](https://square.github.io/pysurvival/metrics/c_index.html) of the `PySurvival` package, "*the C-index is a generalization of the area under the ROC curve (AUC) that can take into account censored data. It represents the global assessment of the model discrimination power: this is the model’s ability to correctly provide a reliable ranking of the survival times based on the individual risk scores*".