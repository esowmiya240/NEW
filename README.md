# Automatic News Headline Generation using NLP

## 1. Project Overview

This mini-project is a web application that automatically generates short and meaningful English news headlines from a long news article.

The application uses:
- Python
- Streamlit
- Hugging Face Transformers
- FLAN-T5 pretrained text-to-text generation model
- PyTorch

## 2. Main Features

- Simple and attractive Streamlit interface
- Long article input
- Optional reference image input
- Word count
- Empty input validation
- Generates 5 alternative headlines
- English, Tamil, and Hindi headline output
- Copy button for each headline
- Clear button
- Project information section

## 3. Folder Structure

```text
Automatic_News_Headline_Generation_NLP/
│
├── app.py
├── requirements.txt
├── README.md
└── sample_input.txt
```

## 4. Installation

### Step 1: Install Python

Use Python 3.9 or later.

### Step 2: Open terminal

Go inside the project folder:

```bash
cd Automatic_News_Headline_Generation_NLP
```

### Step 3: Create a virtual environment (recommended)

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install packages

```bash
pip install -r requirements.txt
```

The first use downloads the required FLAN-T5, BLIP image-captioning, and NLLB translation models from Hugging Face. An internet connection is required initially. BLIP and NLLB are loaded only when an image or a non-English output language is selected.

## 5. Run the Application

```bash
streamlit run app.py
```

A browser window will open with the application.

## 6. Sample Input

See `sample_input.txt`.

Example:

```text
The city administration announced a new public transport program on Monday.
The program will introduce 50 electric buses across major routes over the next
six months. Officials said the project is intended to reduce traffic congestion
and air pollution while improving public transportation for residents. The first
phase will cover routes connecting residential areas with schools, hospitals and
major business centers.
```

## 7. Sample Output

Possible outputs:

1. City to Introduce 50 Electric Buses
2. New Electric Bus Program Aims to Cut Pollution
3. City Launches Public Transport Upgrade
4. 50 Electric Buses Planned for Major Routes
5. New Transit Program Targets Traffic and Pollution

Exact wording can vary because the model generates alternative outputs.

## 8. Module Explanation

### app.py

This is the main application file.

It contains:
- Streamlit user interface
- Model loading
- Article validation
- Word counting
- Tokenization
- Headline generation
- Output display
- Copy buttons
- Clear button

### requirements.txt

Contains all Python libraries required by the project.

### sample_input.txt

Contains a sample news article for testing.

## 9. NLP Model

The project uses `google/flan-t5-base` to generate headlines, `Salesforce/blip-image-captioning-base` to add an uploaded image description to the news context, and `facebook/nllb-200-distilled-600M` to translate headlines into Tamil or Hindi.

FLAN-T5 is a pretrained text-to-text Transformer model. The article is given as text, and the model produces a headline as text.

Basic flow:

```text
Article
   ↓
Tokenizer
   ↓
FLAN-T5
   ↓
Generated Text
   ↓
News Headline
```

## 10. College Demonstration Explanation

"The user enters a long news article in the Streamlit interface. The application first validates the input and calculates the word count. The article is then converted into tokens using the FLAN-T5 tokenizer. The pretrained FLAN-T5 model analyzes the main information in the article and generates multiple short headline alternatives. The headlines are displayed on the webpage, and the user can copy any headline."

## 11. Important Note

Headline quality depends on the article and the pretrained model. The model may occasionally produce similar or imperfect headlines, so the user should review the generated headline before publishing it.
