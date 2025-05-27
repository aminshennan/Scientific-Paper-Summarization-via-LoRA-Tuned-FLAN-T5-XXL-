# Scientific AI Summarizer

A complete, end-to-end pipeline that collects scientific research papers, cleans and explores the data, trains state-of-the-art transformer models, and exposes an elegant drag-and-drop web interface to automatically generate concise summaries of scientific articles (PDF, DOCX or plain text).

---

## Table of Contents
1. ️🎯  [Project Goals](#project-goals)
2. 🗂️  [Repository Structure](#repository-structure)
3. ⚡  [Quick Start](#quick-start)
4. 🔬  [Data Pipeline](#data-pipeline)
5. 🤖  [Model Training & Evaluation](#model-training--evaluation)
6. 🌐  [Web Application](#web-application)
7. 🧑‍💻  [Usage Examples](#usage-examples)
8. 🚀  [Deployment](#deployment)
9. 🤝  [Contributing](#contributing)
10. 📜  [License](#license)
11. 📊  [Results](#results)

---

## Project Goals

1. **Automate summarisation** of lengthy research papers to accelerate literature reviews.
2. **Fine-tune transformer models** – the current training notebook demonstrates parameter-efficient LoRA fine-tuning of **FLAN-T5-XXL** for long-form scientific text summarisation.
3. **Provide a no-install UI** so non-technical users can obtain high-quality summaries with a single drag-and-drop.

## Repository Structure

```text
./
├── src/
│   ├── 1 data featching.ipynb      # Downloads metadata & full-text from APIs (arXiv, Kaggle…)
│   ├── 2 final EDA.ipynb           # Exploratory data analysis & cleaning
│   ├── 3 Model training.ipynb      # Fine-tunes transformer models & saves checkpoints
│   ├── 4 Testing.ipynb             # Generates metrics & qualitative comparisons
│   ├── 5 Hosting.ipynb             # Flask app for serving the model via API and ngrok tunneling
│   ├── index.html                  # Stand-alone front-end (drag-and-drop summariser)
│   ├── LoRA/                       # Saved LoRA adapter model files
│   └── data/                       # Raw & processed CSVs (> 1 GB total)
│       ├── cs_papers_api.csv
│       ├── remaining_papers.csv
│       └── final_combined_dataset.csv
├── Poster.pdf                      # Conference poster (design)
├── Final Report.pdf                # Full project report
├── Slides.pptx                     # Presentation slides
└── README.md                       # ← you are here
```

> **Note** Large datasets are checked in for reproducibility. If you would rather download them on-demand, remove the CSVs and follow the steps in `src/1 data featching.ipynb`.

## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```
2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    The notebooks install their specific dependencies. Key libraries used across the project include:
    `pandas`, `requests`, `pdfminer.six`, `PyMuPDF`, `transformers`, `datasets`, `accelerate`, `evaluate`, `peft`, `bitsandbytes`, `loralib`, `jupyterlab`, `flask`, `pyngrok`, `rouge-score`, `tensorboard`, `py7zr`.

    You can typically install these via pip:
    ```bash
    pip install pandas requests pdfminer.six PyMuPDF transformers datasets accelerate evaluate peft bitsandbytes loralib jupyterlab flask pyngrok rouge-score tensorboard py7zr
    ```
    *Note: `5 Hosting.ipynb` contains `!pip install` commands for its specific needs.* 

4.  **Launch Jupyter Lab** to run the notebooks:
    ```bash
    jupyter lab
    ```
    Execute notebooks in `src/` in numerical order (`1` through `5`).

5.  **Access the Web UI:**
    Open `src/index.html` in a modern web browser. This UI will interact with the Flask API started by `src/5 Hosting.ipynb`.

## Data Pipeline

| Notebook                     | Purpose                                                                                                |
| :--------------------------- | :----------------------------------------------------------------------------------------------------- |
| `src/1 data featching.ipynb` | Queries APIs (e.g., arXiv) ➜ downloads titles, abstracts & full-text ➜ stores in `src/data/`.         |
| `src/2 final EDA.ipynb`      | Deduplicates, cleans LaTeX markup, tokenises & analyses text length statistics.                         |

The cleaned dataset (~200k rows) is saved as `src/data/final_combined_dataset.csv` and used for model training.

## Model Training & Evaluation

The project fine-tunes the FLAN-T5-XXL model using 🤗 HuggingFace `transformers`, `peft` for LoRA, and `datasets`.
Key steps (see `src/3 Model training.ipynb`):

1.  Tokenisation suitable for the FLAN-T5 architecture.
2.  Parameter-Efficient Fine-Tuning (PEFT) with LoRA (`rank=16`, `alpha=32`).
3.  Mixed-precision training on GPU (requires substantial VRAM, e.g., NVIDIA L4).
4.  Evaluation using ROUGE scores.
5.  Saving the trained LoRA adapter in `src/LoRA/`.

`src/4 Testing.ipynb` loads the base model and the trained LoRA adapter to produce quantitative metrics and human-readable summary comparisons against the test set.

## Web Application

The responsive front-end lives in **`src/index.html`** and includes:

*   Drag-and-drop area supporting **TXT, PDF & DOCX** files.
*   Real-time word-count display and dark/light theme toggle.
*   Copy-to-clipboard and "Download as TXT" export buttons.

The page interacts with a `/summarize` REST endpoint provided by a Flask application.

**Backend Service (`src/5 Hosting.ipynb`):**

*   Loads the pre-trained `google/flan-t5-xxl` model and the fine-tuned LoRA adapter from `src/LoRA/`.
*   Uses `Flask` to create a web server.
*   Exposes a `/summarize` endpoint that accepts `POST { "text": "..." }` and returns `{ "summary": "..." }`.
*   Integrates `pyngrok` to create a public tunnel to the local Flask app, making it accessible over the internet (useful for Colab environments or temporary demos).

The core summarization logic in the Flask app:
```python
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel

# ... (model and tokenizer loading as in the notebook) ...

app = Flask(__name__)

# Load base model and tokenizer
model_id = "google/flan-t5-xxl"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    load_in_8bit=True, # For memory efficiency
    device_map="auto"
)

# Load the LoRA adapter
peft_model_path = "src/LoRA" # Assuming adapter is saved here relative to notebook execution
model = PeftModel.from_pretrained(base_model, peft_model_path)
model.eval() # Set model to evaluation mode

# Summarization pipeline (can also be done manually with model.generate)
# summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device_map="auto")

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    text_to_summarize = data.get('text', '')
    if not text_to_summarize:
        return jsonify({'error': 'No text provided'}), 400

    # Manual generation (as per notebook, customizable)
    prompt = f"summrize: {text_to_summarize}"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)
    outputs = model.generate(input_ids=input_ids, max_new_tokens=350, min_length=50, do_sample=True, top_p=0.9)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'summary': summary})

# Ngrok tunneling and app run is typically handled in the notebook cells
# For a standalone app.py, you would add:
# if __name__ == '__main__':
#     # from pyngrok import ngrok
#     # public_url = ngrok.connect(5000)
#     # print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")
#     app.run(port=5000)
```
To run locally (without Ngrok, assuming `app.py` created from the notebook):
```bash
# Ensure all dependencies from 5 Hosting.ipynb are installed
# pip install flask pyngrok transformers torch peft bitsandbytes accelerate
python app.py # (if you save the Flask part as app.py)
```
Then open `src/index.html` and ensure its JavaScript fetch URL points to `http://127.0.0.1:5000/summarize`.

## Usage Examples

1.  **Interactive Summarization:**
    *   Run `src/5 Hosting.ipynb` fully (or the `app.py` derived from it).
    *   Open `src/index.html` in your browser.
    *   Drag a research paper (TXT, PDF, DOCX) onto the drop zone or paste text.
    *   Click *Summarize* to get the ~150-word summary.

2.  **Batch Evaluation:**
    *   Execute `src/4 Testing.ipynb` to compute ROUGE scores on the reserved test set using the trained model.

## Deployment

The application consists of the static `src/index.html` front-end and the Flask backend (derived from `src/5 Hosting.ipynb`).

### 1. Local Demo

   a.  **Start the Flask Backend:**
       Extract the Flask app code from `src/5 Hosting.ipynb` into an `app.py` file in the root directory.
       ```bash
       # From your project root, after creating venv and installing dependencies:
       python app.py
       ```
       This will typically start the Flask server on `http://127.0.0.1:5000`.

   b.  **Serve the HTML Front-end:**
       Open `src/index.html` directly in your browser. Ensure the `fetch` URL in its JavaScript points to `http://127.0.0.1:5000/summarize`.
       Alternatively, serve it with a simple HTTP server from the root directory:
       ```bash
       # In another terminal, from the project root:
       python -m http.server 8080 
       ```
       Then access the UI at `http://localhost:8080/src/index.html`.

### 2. Single-Container (Docker)

   Create an `app.py` from `src/5 Hosting.ipynb`. Then, create a `Dockerfile` in the project root:

   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   COPY requirements.txt .
   # Assuming 5 Hosting.ipynb's pip installs are in requirements.txt or listed here
   RUN pip install --no-cache-dir -r requirements.txt 
   # Or, list them: RUN pip install flask pyngrok transformers torch peft bitsandbytes accelerate sentencepiece protobuf

   COPY src/ ./src/
   COPY app.py ./
   COPY src/LoRA/ ./src/LoRA/ # Critical: Copy the LoRA model files

   EXPOSE 5000
   CMD ["python", "app.py"] # Or use gunicorn: ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
   ```
   *Ensure `requirements.txt` includes `Flask`, `gunicorn` (if used), `transformers`, `torch`, `peft`, `bitsandbytes`, `accelerate`, `sentencepiece`, `protobuf` and any other core dependencies from the notebooks.*
   *The `app.py` should be modified to load models from within the container (e.g., `src/LoRA/`).*

   Build & Run:
   ```bash
   docker build -t sci-summarizer .
   docker run -p 5000:5000 -v $(pwd)/src/data:/app/src/data sci-summarizer
   ```
   The UI (`src/index.html`, potentially served by Flask if `app.py` is configured, or separately) would then call `http://localhost:5000/summarize`.

### 3. Hugging Face Spaces (Recommended for Demo)

1.  **Structure your Space:**
    *   `app.py`: Your Flask application (adapted from `5 Hosting.ipynb`). Make sure model paths are relative within the Space repo.
    *   `requirements.txt`: List all Python dependencies.
    *   `src/index.html`: Your static front-end.
    *   `src/LoRA/`: Your trained LoRA adapter files.
    *   (Optional) `Dockerfile`: If you need more control than `requirements.txt` provides.
2.  **Modify `app.py` to serve `index.html`:**
    ```python
    from flask import send_from_directory

    @app.route("/")
    def root():
        return send_from_directory("src", "index.html") # Assuming index.html is in a 'src' subdir in the Space
    ```
3.  Create a new Space on Hugging Face, choose the SDK (likely Docker or Python with `requirements.txt`).
4.  Upload your files. The Space will build and deploy. The UI and API will be accessible via the Space URL.

## Contributing

Pull requests are welcome! Please open an issue first to discuss major changes. Ensure code is formatted (e.g., with Black) and linted (e.g., with Flake8) before committing.

## License

All code and notebooks are provided **for academic and research purposes only**. Please cite appropriately if you build upon this work.

## Results

(As per `src/4 Testing.ipynb` - ensure these are up-to-date with your latest run)

| Metric     | Base FLAN-T5-XXL | LoRA-tuned FLAN-T5-XXL | Δ (Improvement) |
| :--------- | :--------------- | :--------------------- | :-------------- |
| ROUGE-1    | 28.58            | **31.20**              | +2.62           |
| ROUGE-2    | 7.48             | **8.62**               | +1.14           |
| ROUGE-L    | 16.83            | **17.56**              | +0.73           |
| ROUGE-Lsum | 16.76            | **17.49**              | +0.73           | 