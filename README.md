# Multilingual Text Summarizer

📝 **Multilingual Text Summarizer** is a web application that summarizes text, PDFs, and images in multiple languages using a T5 transformer model. The application is built with Streamlit, EasyOCR, and Hugging Face Transformers.

## Features

- Summarize text input directly
- Summarize content from uploaded PDF, TXT, and image files
- Detect and handle multiple languages
- Translate summarized text to English
- Chat-like prompt system for refining summaries

## Demo

You can try the live demo on [Streamlit Cloud](https://llm-based-text-summarizer.streamlit.app/).

## Screenshots

![App Screenshot](https://github.com/RobinMillford/LLM-Based-Text-Summarizer/blob/main/LLM1.png)
![App Screenshot](https://github.com/RobinMillford/LLM-Based-Text-Summarizer/blob/main/LLM2.png)

## Installation

### Local Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/RobinMillford/llm-based-text-summarizer.git
    cd LLM-Based-Text-Summarizer
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv myenv
    source myenv/bin/activate   # On Windows use `myenv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**

    ```sh
    streamlit run app.py
    ```

### Docker

1. **Pull the Docker image:**

    ```sh
    docker pull yamin69/summarizer:latest
    ```

2. **Run the Docker container:**

    ```sh
    docker run -p 8501:8501 yamin69/summarizer:latest
    ```

## Usage

1. **Navigate to the app URL.**
2. **Choose an input method:**
    - Direct Text Input
    - Upload File (PDF, TXT, Image)
3. **Enter or upload your content.**
4. **Optionally add prompts to refine the summary.**
5. **Click "Generate Summary" to get the summarized text.**

## Contributing

1. **Fork the repository:**

    ```sh
    git fork https://github.com/RobinMillford/llm-based-text-summarizer.git
    ```

2. **Create a branch:**

    ```sh
    git checkout -b feature-branch
    ```

3. **Make your changes and commit them:**

    ```sh
    git commit -am 'Add new feature'
    ```

4. **Push to the branch:**

    ```sh
    git push origin feature-branch
    ```

5. **Create a new Pull Request.**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io)
- [Hugging Face](https://huggingface.co)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

---

For any issues, please create a new [issue](https://github.com/RobinMillford/LLM-Based-Text-Summarizer/issues).
