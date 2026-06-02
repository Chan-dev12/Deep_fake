# 🔍 DeepFake-Detect

> **Can a computer tell if a face in a photo is REAL or FAKE?**
> YES! That's exactly what this project does. 🎉

---

## 🤔 What is a Deepfake?

Imagine someone used a computer to put YOUR face on someone else's body in a video — without your permission. That's called a **deepfake**. They look very real, but they are 100% fake.

This project is a **Deepfake Detector** — it's like a smart detective 🕵️ that looks at a photo of a face and says:
- ✅ **"This face looks REAL!"**
- ❌ **"This face looks FAKE / SYNTHETIC!"**

---

## ✨ How Does It Work? (Simple Version)

Think of it like training a dog 🐶 to sniff out fake money:

1. 📹 **Show it lots of videos** — both real faces and fake faces
2. ✂️ **Crop out just the faces** from every video frame
3. ⚖️ **Balance the dataset** — equal real and fake faces for fair learning
4. 🧠 **Teach the AI** — it studies all the faces until it can tell them apart
5. 🌐 **Use the web app** — upload any photo and get an instant answer!

---

## 🗂️ File Structure — What's Inside This Project?

```
DeepFake-Detect/
│
├── 📄 README.md                          ← You are reading this file!
├── 📄 requirements.txt                   ← List of tools the project needs
├── 📄 LICENSE                            ← Legal info (MIT open source)
│
├── 🎬 00-convert_video_to_image.py       ← STEP 1: Breaks videos into pictures
├── ✂️  01a-crop_faces_with_mtcnn.py      ← STEP 2a: Cuts out just the faces
├── ☁️  01b-crop_faces_with_azure-...py   ← STEP 2b: Face cropping using Azure cloud
├── ⚖️  02-prepare_fake_real_dataset.py   ← STEP 3: Balances real vs fake images
├── 🧠 03-train_cnn.py                    ← STEP 4: Teaches the AI to detect fakes
│
├── 🌐 app.py                             ← The web app server (run this to use it!)
│
├── 📁 templates/
│   └── index.html                        ← The webpage you see in the browser
│
├── 📁 static/
│   ├── style.css                         ← Makes the webpage look beautiful
│   ├── app.js                            ← Makes the webpage interactive
│   └── temp/                             ← Stores temporary result images
│
├── 📁 train_sample_videos/               ← Sample videos + metadata.json
├── 📁 prepared_dataset/                  ← Balanced real & fake face images
├── 📁 split_dataset/                     ← Train / Validation / Test folders
├── 📁 tmp_fake_faces/                    ← Temporary fake faces during processing
├── 📁 tmp_checkpoint/
│   └── best_model.h5                     ← 🏆 The trained AI brain (saved here!)
├── 📁 tmp_debug/                         ← Debug files during training
└── 📁 img/                               ← Screenshots used in this README
```

---

## 🚀 How to Use This Project (Step by Step)

### ✅ Step 0 — Install Requirements

Make sure you have **Python 3** installed. Then open a terminal and run:

```bash
pip install -r requirements.txt
```

This installs all the tools the project needs (like TensorFlow, OpenCV, MTCNN).

---

### 🎬 Step 1 — Extract Frames from Videos

```bash
python 00-convert_video_to_image.py
```

📌 **What it does:** Takes every video in the `train_sample_videos/` folder and saves each frame as a separate image (like taking screenshots every second).

🖼️ It also automatically resizes images based on their size for best performance.

---

### ✂️ Step 2 — Crop the Faces

```bash
python 01a-crop_faces_with_mtcnn.py
```

📌 **What it does:** Looks at each frame image and finds the face in it. Then it zooms in and saves just the face (with a 30% border around it for context).

> **Optional:** If you don't have a good GPU, you can use `01b-crop_faces_with_azure-vision-api.py` instead — it uses Microsoft's cloud service. You'll need to add your own API key first.

---

### ⚖️ Step 3 — Prepare the Dataset

```bash
python 02-prepare_fake_real_dataset.py
```

📌 **What it does:** 
- Collects all the real faces and fake faces
- Makes sure there are **equal numbers of real and fake** (so the AI doesn't get biased)
- Splits them into 3 groups:
  - **80% Training** — the AI studies these
  - **10% Validation** — used to check learning during training
  - **10% Testing** — used to test the final AI at the end

---

### 🧠 Step 4 — Train the AI

```bash
python 03-train_cnn.py
```

📌 **What it does:** This is where the magic happens! 🎩 It teaches the **EfficientNet-B0** neural network to recognize deepfake faces. It:
- Runs for up to **20 rounds** (called epochs)
- Automatically stops early if it's not getting better
- Saves the best version of the AI to `tmp_checkpoint/best_model.h5`

> ⚠️ **Important:** The more training data you have, the smarter the AI becomes! The 5 sample videos included are just for testing the pipeline — download a real dataset for accurate results.

---

### 🌐 Step 5 — Launch the Web App!

```bash
python app.py
```

📌 **What it does:** Starts a local website on your computer. Open your browser and go to:

## 👉 http://127.0.0.1:5000

You can now:
1. 📤 **Upload any face image** (drag & drop or browse)
2. 🔬 **Wait for analysis** (the AI scans the face in seconds)
3. 🟢 See **REAL / PRISTINE** or 🔴 **SYNTHETIC / FAKE** verdict
4. 📊 View the **confidence score** (how sure the AI is)
5. 🌡️ See the **Forgery Heatmap** — highlights exactly where the face looks suspicious!

---

## 🌡️ What is the Forgery Heatmap?

After the analysis, you'll see **two images side by side**:

| Original Face | Forgery Map |
|---|---|
| The face as cropped by the AI | A thermal overlay showing suspicious zones |
| 🧍 Normal face photo | 🔴 Red = highly suspicious area |

**How to read the heatmap:**
- 🔴 **Red/Orange = HOT zones** → The AI found something suspicious here
- 🔵 **Blue/Green = COLD zones** → These areas look normal
- If the verdict is FAKE, the red zones show WHERE the face was digitally altered!

---

## 🧠 What AI Model Does It Use?

This project uses **EfficientNet-B0** — a state-of-the-art neural network originally trained on millions of images from the internet (ImageNet). We then fine-tune it to focus on deepfake detection.

The architecture:
```
Input Image (128x128 pixels)
        ↓
EfficientNet-B0 Backbone (feature extractor)
        ↓
Global Max Pooling
        ↓
Dense Layer (512 neurons) → ReLU → Dropout
        ↓
Dense Layer (128 neurons) → ReLU
        ↓
Output: Single value between 0.0 and 1.0
  → Close to 1.0 = REAL ✅
  → Close to 0.0 = FAKE ❌
```

---

## 📦 Training Datasets

For accurate detection, train on these public deepfake datasets:

| Dataset | Description |
|---|---|
| [DeepFake-TIMIT](https://www.idiap.ch/dataset/deepfaketimit) | Early deepfake benchmark |
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | ~1000 videos, multiple methods |
| [Google DFD](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html) | Google's deepfake dataset |
| [Celeb-DF](https://github.com/danmohaha/celeb-deepfakeforensics) | Celebrity deepfakes |
| [Facebook DFDC](https://ai.facebook.com/datasets/dfdc/) | Facebook's 100K+ videos |

---

## ❓ Common Questions

**Q: Why does it say REAL for all my test images?**
> The model included was trained on only 5 sample videos (very tiny!). Download and use a large real dataset to get accurate predictions.

**Q: Can I upload any photo from the internet?**
> Yes! Just go to `http://127.0.0.1:5000`, upload the image, and get your result instantly.

**Q: How long does training take?**
> On a CPU without a GPU, it can take hours for large datasets. Using a GPU (via WSL2 on Windows) speeds it up significantly.

**Q: What image types are supported?**
> JPEG, PNG, and WEBP are all supported.

---

## 🛠️ Built With

| Tool | What it Does |
|---|---|
| Python 3 | The programming language |
| TensorFlow + Keras | Builds and trains the AI |
| EfficientNet-B0 | The AI brain architecture |
| MTCNN | Detects and crops faces |
| OpenCV | Reads and processes images |
| Flask | Runs the local web server |
| HTML + CSS + JS | Makes the web interface look great |

---

## 📝 License

This project is open source under the **MIT License** — you can use it freely for learning, research, or building your own tools!

---


*Made with ❤️ to help fight misinformation and deepfake content.*
