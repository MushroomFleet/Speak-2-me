# Speak2me_v1
![demo-ui](https://raw.githubusercontent.com/MushroomFleet/Speak-2-me/refs/heads/main/images/demo-ui.png)

A Gradio-based speech-to-prompt pipeline integrating transcription (via Whisper), natural language generation (via Ollama), and text-to-speech (via Kokoro TTS). This application automatically downloads the Whisper Tiny model by default, with support for the Medium model and CPU fallback.

## Features
- 🎤 Speech transcription using Whisper.
- 🤖 AI text generation powered by Ollama.
- 🔊 TTS synthesis using Kokoro TTS.
- 📋 Automatic clipboard copying of transcriptions.
- ⚙️ Configurable Whisper model (Tiny/Medium) with CPU mode toggle.

## Installation

1. **Clone the Repository**
   - Clone the repository from GitHub:
     ```bash
     git clone https://github.com/MushroomFleet/Speak-2-me
     cd Speak2me_v1
     ```

2. **Install Dependencies**
   - Install Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Ensure that PyTorch is installed with GPU support if available (optional).

3. **Model Files**
   - **Kokoro TTS Models:**  
     Place the required KokoroTTS model files in the root directory of the project.  
     - **Model link:** [Kokoro-v1.0.onnx](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx), [voices-v1.0.bin](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin).

4. **Whisper Model**
   - The *Whisper Tiny* model is automatically downloaded on first run.  
   - You can switch to the *Whisper Medium* model for higher accuracy via the app settings.
   - CPU fallback is supported if GPU is not available.

## Usage

1. **Run the Application**
   - Launch the Gradio interface:
     ```bash
     python Speak2me_v1.py
     ```
   - Alternatively, you can run the provided batch file:
     ```bash
     run-gradio.bat
     ```

2. **Using the App**
   - **Speech to Prompt Tab:**  
     Record or upload your audio, then click the 🎯 **Transcribe** button to convert speech to text. The transcription will be saved to an output file and automatically copied to your clipboard.
     
   - **Ollama Tab:**  
     Configure your Ollama settings by providing the URL, selecting an AI model, and setting up the system prompt. This will generate an AI response based on your transcription.
     
   - **Response to Speech Tab:**  
     Adjust the text-to-speech settings (speed and voice) to synthesize the audio output from the AI response. The voice menu now includes new options for Indian and Spanish voices.

## Notes
- The app defaults to the **Whisper Tiny** model for faster performance.  
- Switch to the **Whisper Medium** model for improved transcription accuracy if needed.
- CPU fallback is enabled for systems lacking GPU support.
- Ensure that the KokoroTTS model files are present in the root directory before running the application.

Happy coding! 🚀
