# 🛰️ Satellite Building Change Detection

A deep learning system for detecting building changes between bi-temporal satellite images using a custom **Siamese FC-Siam-Diff** architecture trained on the **LEVIR-CD** dataset.

The application takes a **Before** and **After** satellite image and produces a pixel-level change map identifying areas where building changes are detected.

🔗 **GitHub:** https://github.com/Vijayendra2707/satellite-building-change-detection

🚀 **Live Demo:** https://satellite-building-change-detection-msmjsy4mxited5j7nuqkfw.streamlit.app/

🤗 **Hugging Face Model:** https://huggingface.co/VJ2707/satellite-building-change-detection

---

## 📌 Overview

Satellite imagery captured at different points in time can be used to identify changes in buildings and urban areas.

This project implements a deep learning-based **bi-temporal change detection pipeline**:

```text
Before Satellite Image
          +
After Satellite Image
          ↓
    Siamese Encoder
          ↓
Feature Extraction
          ↓
Absolute Feature Difference
          ↓
Decoder + Skip Connections
          ↓
Pixel-wise Change Probability
          ↓
Threshold = 0.70
          ↓
Binary Change Mask 

```
The model performs pixel-level binary segmentation, where each pixel is classified as either:

0 → No building change
1 → Building change
🧠 Model Architecture

The project uses a custom Siamese FC-Siam-Diff architecture.

Two satellite images are passed through the same encoder with shared weights.

                 BEFORE IMAGE
                      │
                      ▼
              ┌──────────────┐
              │ Shared CNN   │
              │   Encoder    │
              └──────────────┘
                      │
                  Features
                      │
                      │
              | Feature A - Feature B |
                      │
                      ▼
              Feature Differences
                      │
                      ▼
              ┌──────────────┐
              │   Decoder    │
              │ + Skip Conn. │
              └──────────────┘
                      │
                      ▼
              Change Probability
                      │
                      ▼
                Binary Mask




                 AFTER IMAGE
                      │
                      ▼
              ┌──────────────┐
              │ Shared CNN   │
              │   Encoder    │
              └──────────────┘
                      │
                  Features
Encoder

The Siamese encoder consists of four convolutional stages:

RGB
 ↓
3 → 64
 ↓
64 → 128
 ↓
128 → 256
 ↓
256 → 512

Each convolutional block uses:

3×3 Convolution
Batch Normalization
ReLU
3×3 Convolution
Batch Normalization
ReLU

Max pooling is used between encoder stages.

Feature Difference

At each corresponding encoder level:

Difference = |Feature_Before - Feature_After|

This allows the network to learn spatial differences between the two temporal observations.

Decoder

The decoder reconstructs the spatial resolution using feature differences and skip connections.

768 → 256
384 → 128
192 → 64
64 → 1

The final 1×1 convolution produces a single-channel pixel-wise change probability map.

📊 Dataset

The model was trained and evaluated using the LEVIR-CD building change detection dataset.

LEVIR-CD contains pairs of high-resolution satellite images captured at different points in time together with binary building-change annotations.

The dataset is not included in this repository.

📈 Model Performance
Patch-Level / Pixel-Level Evaluation

The model achieved the following test performance:

Metric	Score
IoU	0.7691
F1 Score	0.8695
Precision	0.8809
Recall	0.8583
Inference Threshold
0.70

The threshold is applied to the predicted probability map to generate the final binary change mask.

Full-Image Evaluation

The system also supports reconstruction of predictions for full-resolution satellite images.

Metric	Score
Mean IoU	0.6574
Mean F1	0.7495
Mean Precision	0.7288
Mean Recall	0.7252

Evaluation was performed on 128 full-resolution test images.

🖥️ Live Application

The project is deployed using Streamlit and can be accessed directly through the live demo.

Application

https://satellite-building-change-detection-msmjsy4mxited5j7nuqkfw.streamlit.app/

Users can:

Upload a Before satellite image
Upload an After satellite image
Run building change detection
View the predicted change mask
View pixel-wise change probabilities
View detected changes overlaid on the satellite image
Analyze the number and percentage of changed pixels
🎥 Application Screenshots
Application Interface

The application provides a simple interface for uploading the two temporal satellite images.

Input Satellite Images

The model receives two images representing the same geographical region at different time periods.

Change Probability Map

The probability map represents the model's confidence that each pixel belongs to a changed region.

Higher probability values indicate stronger model confidence.

Predicted Change Mask

The probability map is thresholded at 0.70 to generate the final binary change mask.

Change Overlay

Detected building changes are highlighted directly on the satellite imagery for easier visual interpretation.

🔍 Inference Pipeline

The model operates on 256 × 256 image patches.

For a full 1024 × 1024 satellite image:

1024 × 1024 Before Image
            +
1024 × 1024 After Image
            │
            ▼
      256 × 256 Patches
            │
            ▼
      Siamese CNN Encoder
            │
            ▼
     Feature Differences
            │
            ▼
          Decoder
            │
            ▼
    Patch Probability Maps
            │
            ▼
    Full Image Reconstruction
            │
            ▼
    1024 × 1024 Probability Map
            │
            ▼
       Threshold = 0.70
            │
            ▼
      Binary Change Mask

The inference pipeline can also process individual 256 × 256 patches.

📊 Example Detection

For an example test patch, the system produced:

Changed Pixels : 18,228
Changed Area   : 27.81%
Threshold      : 0.70

The output includes:

Pixel-wise probability map
Binary change mask
Change overlay
Detection summary

# 📁 Project Structure

```text
satellite-building-change-detection/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── model.py
│   └── inference.py
│
├── checkpoints/
│   └── best_model.pth
│
├── results/
│   ├── figures/
│   │   ├── best_change_test_102_x256_y256.png
│   │   ├── worst_change_test_100_x256_y0.png
│   │   └── high_fp_test_103_x768_y768.png
│   │
│   ├── inference/
│   │   └── test_69_change_mask.png
│   │
│   ├── final_metrics.json
│   └── full_image_test_results.csv
│
├── assets/
│   ├── app_home.png
│   ├── input_images.png
│   ├── probability_map.png
│   ├── change_mask.png
│   └── change_overlay.png
│
├── .gitignore
├── .gitattributes
├── README.md
└── requirements.txt
```
🛠️ Technologies
Python
PyTorch
NumPy
Pillow
Matplotlib
Streamlit
Hugging Face Hub
Git
GitHub
Git LFS
🎯 Key Technical Concepts
Siamese Neural Networks
FC-Siam-Diff
Bi-temporal Satellite Image Analysis
Building Change Detection
Pixel-level Binary Segmentation
Feature Difference Learning
Encoder-Decoder Architecture
Skip Connections
Patch-based Inference
Full-resolution Image Reconstruction
Probability Thresholding
IoU
F1 Score
Precision
Recall
🔬 Evaluation

The system was evaluated at both patch and full-image levels.

Patch-level evaluation measures the quality of individual 256 × 256 predictions.

Full-image evaluation reconstructs the predicted patches into the original 1024 × 1024 image resolution and evaluates the complete change map.

The final pixel-level aggregate results were:

IoU       : 0.7691
F1        : 0.8695
Precision : 0.8809
Recall    : 0.8583

The project also includes qualitative analysis of:

Best change detection cases
Difficult / worst cases
False-positive cases
No-change cases
🚧 Limitations

The model is specifically trained for building change detection using the LEVIR-CD dataset.

It is not a general-purpose satellite object detector.

The system does not classify changes into categories such as:

Roads
Forests
Rivers
Vehicles
Other land-cover classes

Instead, the model learns patterns corresponding to building changes from the training data.

🔮 Future Improvements

Potential improvements include:

Improving performance on difficult change cases
Reducing false-positive predictions
Experimenting with stronger augmentation strategies
Adding morphological/post-processing techniques
Evaluating additional change detection architectures
Testing additional satellite change detection datasets
GPU-backed deployment
Model optimization for faster inference
Improving full-resolution reconstruction
Adding confidence-based visualization
Adding batch inference support
🚀 Running Locally

Clone the repository:

git clone https://github.com/Vijayendra2707/satellite-building-change-detection.git
cd satellite-building-change-detection

Install the required dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app/streamlit_app.py

The application will be available at:

http://localhost:8501
🤗 Model

The trained model checkpoint is hosted separately on Hugging Face to make deployment easier and avoid storing the large model directly in the Streamlit application repository.

Hugging Face Model:

https://huggingface.co/VJ2707/satellite-building-change-detection

The Streamlit application downloads the trained checkpoint from Hugging Face during inference.

📌 Project Highlights
Custom Siamese FC-Siam-Diff architecture
Shared-weight bi-temporal feature extraction
Multi-scale feature difference learning
Encoder-decoder segmentation architecture
Skip connections
Patch-based inference
Full-resolution image reconstruction
Pixel-level building change detection
Interactive Streamlit application
Hugging Face model hosting
Quantitative evaluation using IoU, F1, Precision and Recall
Qualitative false-positive and difficult-case analysis
Cloud deployment using Streamlit
👨‍💻 Author

Vijayendra Rane

B.Tech Computer Science Engineering

📌 Pune, India

📄 License

This project is intended for educational, research, and portfolio purposes.



### One important change I'd make from your old README


Don't put the **90+ MB model checkpoint** directly in the README or rely on GitHub LFS for Streamlit deployment. Your current setup is better:


**GitHub**
→ code + README + results + screenshots


**Hugging Face**
→ trained model


**Streamlit Cloud**
→ deployed application


That's a much cleaner portfolio architecture.


And your screenshots are actually worth putting in the README — especially the **application UI + probability map + mask + overlay**. They immediately show a recruiter that this isn't just a notebook/model sitting on GitHub. 














