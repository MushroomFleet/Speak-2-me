#!/usr/bin/env python3
import os
import json
import re
import subprocess
from datetime import datetime
import gradio as gr
import torch
import whisper
import numpy as np
import importlib.util
try:
    from ollama import Client
except ImportError:
    raise ImportError("ollama module not found. Please install it for Ollama integration.")
import pyperclip

# -------------------------------
# Whisper Transcriber Class (mirroring Careless-Whisper functionality)
# -------------------------------
class WhisperTranscriber:
    def __init__(self):
        self.force_cpu = self.load_saved_cpu_choice()
        self.device = "cpu" if self.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.model = None
        self.current_model_name = self.load_saved_model_choice()
        self.available_models = {
            "tiny": {
                "name": "Tiny",
                "description": "Tiny model - Fastest, lowest accuracy (about 1GB)",
                "size": "~1GB"
            },
            "medium": {
                "name": "Medium",
                "description": "Medium model - Good balance of speed/accuracy (about 5GB)",
                "size": "~5GB"
            }
        }
    
    def load_saved_model_choice(self):
        try:
            if os.path.exists('model_config.json'):
                with open('model_config.json', 'r') as f:
                    config = json.load(f)
                    return config.get('model', 'tiny')
        except Exception as e:
            print(f"Error loading model config: {e}")
        return 'tiny'
    
    def load_saved_cpu_choice(self):
        try:
            if os.path.exists('model_config.json'):
                with open('model_config.json', 'r') as f:
                    config = json.load(f)
                    return config.get('force_cpu', False)
        except Exception as e:
            print(f"Error loading CPU config: {e}")
        return False
    
    def save_config(self):
        try:
            config = {
                'model': self.current_model_name,
                'force_cpu': self.force_cpu
            }
            with open('model_config.json', 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def set_device_mode(self, force_cpu):
        if self.force_cpu != force_cpu:
            self.force_cpu = force_cpu
            self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Switching to device: {self.device}")
            if self.model is not None:
                self.model = None
                torch.cuda.empty_cache()
            self.save_config()
            return f"Switched to {self.device.upper()} mode"
        return f"Already in {self.device.upper()} mode"
    
    def load_model(self, model_name=None):
        if model_name is None:
            model_name = self.current_model_name
        try:
            if self.model is None or model_name != self.current_model_name:
                print(f"Loading Whisper {model_name} model...")
                self.model = whisper.load_model(model_name)
                if self.device == "cuda":
                    self.model = self.model.cuda()
                    torch.cuda.empty_cache()
                    print("Model loaded on GPU")
                else:
                    self.model = self.model.cpu()
                    print("Model loaded on CPU")
                self.current_model_name = model_name
                self.save_config()
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def get_current_model_info(self):
        info = self.available_models.get(self.current_model_name, {})
        status = "Loaded" if self.model is not None else "Not Loaded"
        return {
            "name": info.get("name", self.current_model_name),
            "description": info.get("description", ""),
            "status": status,
            "device": self.device.upper()
        }
    
    def change_model(self, model_name):
        if model_name not in self.available_models:
            return f"Error: Invalid model selection '{model_name}'"
        try:
            self.load_model(model_name)
            return f"Successfully switched to {self.available_models[model_name]['name']} model"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def preprocess_audio(self, audio):
        try:
            if audio is None:
                raise ValueError("No audio data received")
            if isinstance(audio, tuple):
                sample_rate, audio_data = audio
                if audio_data is None or len(audio_data) == 0:
                    raise ValueError("Empty audio data received")
                if isinstance(audio_data, np.ndarray):
                    audio_data = audio_data.astype(np.float32)
                    max_val = np.abs(audio_data).max()
                    if max_val > 1.0:
                        audio_data = audio_data / max_val
                    return audio_data
            if isinstance(audio, str):
                if not os.path.exists(audio):
                    raise ValueError(f"Audio file not found: {audio}")
                return audio
            raise ValueError(f"Unsupported audio format: {type(audio)}")
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None
    
    def transcribe_audio(self, audio):
        try:
            model = self.load_model()
            if model is None:
                return "Error: Model failed to load"
            processed_audio = self.preprocess_audio(audio)
            if processed_audio is None:
                return "Error: Failed to process audio input"
            transcribe_options = {
                "fp16": torch.cuda.is_available() and not self.force_cpu,
                "language": "en",
                "task": "transcribe"
            }
            print("Starting transcription...")
            result = model.transcribe(processed_audio, **transcribe_options)
            print("Transcription complete.")
            # Save transcription to file
            base_name = "recording"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"transcription_{base_name}_{timestamp}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
            # Copy to clipboard
            pyperclip.copy(result["text"])
            print(f"Transcription saved to: {output_file}")
            print("Transcription copied to clipboard automatically")
            return result["text"]
        except Exception as e:
            error_msg = f"Error during transcription: {str(e)}"
            print(error_msg)
            return error_msg

# -------------------------------
# Ollama Integration (mirroring Ollama-local-Gradio)
# -------------------------------
def list_ollama_models(url="http://127.0.0.1:11434"):
    try:
        client = Client(host=url)
        models = client.list().get('models', [])
        try:
            return [model['model'] for model in models]
        except:
            return [model['name'] for model in models]
    except Exception as e:
        return [f"Error connecting: {str(e)}"]

def generate_text(prompt, system_prompt, model, temperature=0.8, top_k=40, top_p=0.9,
                  url="http://127.0.0.1:11434", format="text", debug=False):
    try:
        client = Client(host=url)
        options = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
        if debug:
            print(f"""
Request to Ollama:
URL: {url}
Model: {model}
System Prompt: {system_prompt}
Prompt: {prompt}
Options: {options}
            """)
        response = client.generate(
            model=model,
            system=system_prompt,
            prompt=prompt,
            options=options,
            format='' if format=="text" else format
        )
        if debug:
            print("Ollama response:", response)
        return response['response']
    except Exception as e:
        return f"❌ Error: {str(e)}"

# -------------------------------
# Kokoro TTS Integration (mirroring KokoroTTS-Gradio)
# -------------------------------
# Hardcoded voice menu mapping remains unchanged
voice_menu = {
        # US Voices
    "af_alloy (en)": {"name": "af_alloy", "lang": "en-us"},
    "af_aoede (en)": {"name": "af_aoede", "lang": "en-us"},
    "af_bella (en)": {"name": "af_bella", "lang": "en-us"},
    "af_heart (en)": {"name": "af_heart", "lang": "en-us"},
    "af_jessica (en)": {"name": "af_jessica", "lang": "en-us"},
    "af_kore (en)": {"name": "af_kore", "lang": "en-us"},
    "af_nicole (en)": {"name": "af_nicole", "lang": "en-us"},
    "af_nova (en)": {"name": "af_nova", "lang": "en-us"},
    "af_river (en)": {"name": "af_river", "lang": "en-us"},
    "af_sarah (en)": {"name": "af_sarah", "lang": "en-us"},
    "af_sky (en)": {"name": "af_sky", "lang": "en-us"},

    "am_adam (en)": {"name": "am_adam", "lang": "en-us"},
    "am_echo (en)": {"name": "am_echo", "lang": "en-us"},
    "am_eric (en)": {"name": "am_eric", "lang": "en-us"},
    "am_fenrir (en)": {"name": "am_fenrir", "lang": "en-us"},
    "am_liam (en)": {"name": "am_liam", "lang": "en-us"},
    "am_michael (en)": {"name": "am_michael", "lang": "en-us"},
    "am_onyx (en)": {"name": "am_onyx", "lang": "en-us"},
    "am_puck (en)": {"name": "am_puck", "lang": "en-us"},

    # GB Voices
    "bf_alice (en)": {"name": "bf_alice", "lang": "en-gb"},
    "bf_emma (en)": {"name": "bf_emma", "lang": "en-gb"},
    "bf_isabella (en)": {"name": "bf_isabella", "lang": "en-gb"},
    "bf_lily (en)": {"name": "bf_lily", "lang": "en-gb"},
    "bm_daniel (en)": {"name": "bm_daniel", "lang": "en-gb"},
    "bm_fable (en)": {"name": "bm_fable", "lang": "en-gb"},
    "bm_george (en)": {"name": "bm_george", "lang": "en-gb"},
    "bm_lewis (en)": {"name": "bm_lewis", "lang": "en-gb"},

    # FR Voices
    "ff_siwis (fr)": {"name": "ff_siwis", "lang": "fr-fr"},

    # IT Voices
    "if_sara (it)": {"name": "if_sara", "lang": "it"},
    "im_nicola (it)": {"name": "im_nicola", "lang": "it"},

    # JP Voices
    "jf_alpha (ja)": {"name": "jf_alpha", "lang": "ja"},
    "jf_gongitsune (ja)": {"name": "jf_gongitsune", "lang": "ja"},
    "jf_nezumi (ja)": {"name": "jf_nezumi", "lang": "ja"},
    "jf_tebukuro (ja)": {"name": "jf_tebukuro", "lang": "ja"},
    "jm_kumo (ja)": {"name": "jm_kumo", "lang": "ja"},

    # CN Voices
    "zf_xiaobei (cmn)": {"name": "zf_xiaobei", "lang": "cmn"},
    "zf_xiaoni (cmn)": {"name": "zf_xiaoni", "lang": "cmn"},
    "zf_xiaoxiao (cmn)": {"name": "zf_xiaoxiao", "lang": "cmn"},
    "zf_xiaoyi (cmn)": {"name": "zf_xiaoyi", "lang": "cmn"}
}

def tts(text, speed, voice_choice):
    from kokoro_onnx import Kokoro
    import soundfile as sf
    # Inline voice menu for TTS
    voice_menu = {
            # US Voices
    "af_alloy (en)": {"name": "af_alloy", "lang": "en-us"},
    "af_aoede (en)": {"name": "af_aoede", "lang": "en-us"},
    "af_bella (en)": {"name": "af_bella", "lang": "en-us"},
    "af_heart (en)": {"name": "af_heart", "lang": "en-us"},
    "af_jessica (en)": {"name": "af_jessica", "lang": "en-us"},
    "af_kore (en)": {"name": "af_kore", "lang": "en-us"},
    "af_nicole (en)": {"name": "af_nicole", "lang": "en-us"},
    "af_nova (en)": {"name": "af_nova", "lang": "en-us"},
    "af_river (en)": {"name": "af_river", "lang": "en-us"},
    "af_sarah (en)": {"name": "af_sarah", "lang": "en-us"},
    "af_sky (en)": {"name": "af_sky", "lang": "en-us"},

    "am_adam (en)": {"name": "am_adam", "lang": "en-us"},
    "am_echo (en)": {"name": "am_echo", "lang": "en-us"},
    "am_eric (en)": {"name": "am_eric", "lang": "en-us"},
    "am_fenrir (en)": {"name": "am_fenrir", "lang": "en-us"},
    "am_liam (en)": {"name": "am_liam", "lang": "en-us"},
    "am_michael (en)": {"name": "am_michael", "lang": "en-us"},
    "am_onyx (en)": {"name": "am_onyx", "lang": "en-us"},
    "am_puck (en)": {"name": "am_puck", "lang": "en-us"},

    # GB Voices
    "bf_alice (en)": {"name": "bf_alice", "lang": "en-gb"},
    "bf_emma (en)": {"name": "bf_emma", "lang": "en-gb"},
    "bf_isabella (en)": {"name": "bf_isabella", "lang": "en-gb"},
    "bf_lily (en)": {"name": "bf_lily", "lang": "en-gb"},
    "bm_daniel (en)": {"name": "bm_daniel", "lang": "en-gb"},
    "bm_fable (en)": {"name": "bm_fable", "lang": "en-gb"},
    "bm_george (en)": {"name": "bm_george", "lang": "en-gb"},
    "bm_lewis (en)": {"name": "bm_lewis", "lang": "en-gb"},

    # FR Voices
    "ff_siwis (fr)": {"name": "ff_siwis", "lang": "fr-fr"},

    # IT Voices
    "if_sara (it)": {"name": "if_sara", "lang": "it"},
    "im_nicola (it)": {"name": "im_nicola", "lang": "it"},

    # JP Voices
    "jf_alpha (ja)": {"name": "jf_alpha", "lang": "ja"},
    "jf_gongitsune (ja)": {"name": "jf_gongitsune", "lang": "ja"},
    "jf_nezumi (ja)": {"name": "jf_nezumi", "lang": "ja"},
    "jf_tebukuro (ja)": {"name": "jf_tebukuro", "lang": "ja"},
    "jm_kumo (ja)": {"name": "jm_kumo", "lang": "ja"},

    # CN Voices
    "zf_xiaobei (cmn)": {"name": "zf_xiaobei", "lang": "cmn"},
    "zf_xiaoni (cmn)": {"name": "zf_xiaoni", "lang": "cmn"},
    "zf_xiaoxiao (cmn)": {"name": "zf_xiaoxiao", "lang": "cmn"},
    "zf_xiaoyi (cmn)": {"name": "zf_xiaoyi", "lang": "cmn"}
    }
    selected_voice = voice_menu.get(voice_choice)
    if not selected_voice:
        return "Selected voice not found.", None
    voice_name = selected_voice["name"]
    lang = selected_voice["lang"]
    
    try:
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        voice_style = kokoro.get_voice_style(voice_name)
        samples, sample_rate = kokoro.create(text, voice=voice_style, speed=speed, lang=lang)
    except Exception as e:
        return f"Error during TTS generation: {str(e)}", None
    
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join("output", f"audio_{timestamp}.wav")
    try:
        sf.write(output_filename, samples, sample_rate)
    except Exception as e:
        return f"Error saving audio file: {str(e)}", None

    return f"Audio file saved as {output_filename}", output_filename

# -------------------------------
# Speak2me Pipeline: Integrates Whisper -> Ollama -> TTS
# -------------------------------
def speak2me_pipeline(audio, ollama_url, ollama_model, system_prompt, temperature,
                      top_k, top_p, format_val, debug, tts_speed, tts_voice):
    if audio is None:
        return "No audio provided.", "", ""
    # Transcribe audio with Whisper
    transcriber = WhisperTranscriber()
    transcription = transcriber.transcribe_audio(audio)
    # Save transcription to file with timestamp
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.join("output", f"input_{timestamp}.txt")
    with open(input_filename, "w", encoding="utf-8") as f:
        f.write(transcription)
    # Use transcription as prompt for Ollama
    ollama_response = generate_text(transcription, system_prompt, ollama_model, temperature, top_k, top_p, ollama_url, format_val, debug)
    # Save Ollama response to file with timestamp
    prompt_filename = os.path.join("output", f"prompt_{timestamp}.txt")
    with open(prompt_filename, "w", encoding="utf-8") as f:
        f.write(ollama_response)
    # Clean up Ollama response for TTS (remove special characters)
    cleaned_response = re.sub(r'[^A-Za-z0-9\s.,?!]', '', ollama_response)
    # Generate TTS from cleaned response
    tts_status, tts_audio = tts(cleaned_response, tts_speed, tts_voice)
    return transcription, ollama_response, tts_audio

# -------------------------------
# Build Gradio Interface (Combined Standalone App)
# -------------------------------
with gr.Blocks(title="Speak2me_v1") as demo:
    with gr.Tabs():
        with gr.Tab("Speech to Prompt"):
            gr.Markdown("## Record or upload audio for transcription")
            with gr.Row():
                with gr.Column(scale=6):
                    audio_input = gr.Audio(label="Audio Input", type="filepath")
                    transcribe_btn = gr.Button("🎯 Transcribe")
                    transcription_display = gr.TextArea(label="Transcription", interactive=False, lines=5)
                with gr.Column(scale=4):
                    gr.Markdown("### Whisper Model Settings")
                    whisper_model_dropdown = gr.Dropdown(
                        choices=list(WhisperTranscriber().available_models.keys()),
                        value=WhisperTranscriber().current_model_name,
                        label="Select Whisper Model"
                    )
                    cpu_toggle = gr.Checkbox(label="Force CPU Mode", value=WhisperTranscriber().load_saved_cpu_choice())
                    apply_whisper_btn = gr.Button("Apply Changes")
                    whisper_info = gr.Markdown(label="Current Model Info")
            # Handler for transcription
            def transcribe_audio(audio):
                transcriber = WhisperTranscriber()
                return transcriber.transcribe_audio(audio)
            transcribe_btn.click(fn=transcribe_audio, inputs=audio_input, outputs=transcription_display)
            # Handler for Whisper model settings
            def update_whisper(model_choice, force_cpu):
                transcriber = WhisperTranscriber()
                msg = transcriber.change_model(model_choice)
                mode_msg = transcriber.set_device_mode(force_cpu)
                info = transcriber.get_current_model_info()
                info_text = f"""
**Name:** {info['name']}
**Description:** {info['description']}
**Status:** {info['status']}
**Device:** {info['device']}
                """
                return msg + " | " + mode_msg, info_text
            apply_whisper_btn.click(fn=update_whisper, inputs=[whisper_model_dropdown, cpu_toggle], outputs=[gr.Textbox(visible=False), whisper_info])
        
        with gr.Tab("Ollama"):
            gr.Markdown("## Configure Ollama (LLM) Settings")
            with gr.Row():
                with gr.Column():
                    ollama_url = gr.Textbox(label="Ollama URL", value="http://127.0.0.1:11434")
                    # Dropdown for models with refresh functionality
                    ollama_model = gr.Dropdown(
                        label="Ollama Model",
                        choices=list_ollama_models(),
                        value=list_ollama_models()[0] if list_ollama_models() and not list_ollama_models()[0].startswith("Error") else "default-model"
                    )
                    refresh_btn = gr.Button("🔄 Refresh Models")
                    def refresh_models(url):
                        models = list_ollama_models(url)
                        default_val = models[0] if models and not models[0].startswith("Error") else "default-model"
                        return { "choices": models, "value": default_val }
                    refresh_btn.click(fn=refresh_models, inputs=ollama_url, outputs=ollama_model)
                with gr.Column():
                    system_prompt = gr.Textbox(label="System Prompt", value="You are a helpful AI assistant.", lines=2)
            with gr.Row():
                temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, value=0.8, step=0.05)
                top_k = gr.Slider(label="Top K", minimum=0, maximum=100, value=40, step=1)
                top_p = gr.Slider(label="Top P", minimum=0, maximum=1, value=0.9, step=0.05)
            with gr.Row():
                format_val = gr.Radio(label="Output Format", choices=["text", "json"], value="text")
                debug = gr.Checkbox(label="Debug Mode", value=False)
            ollama_response_display = gr.TextArea(label="Ollama Response", interactive=False, lines=5)
        
        with gr.Tab("Response to Speech"):
            gr.Markdown("## Configure TTS Settings")
            with gr.Row():
                tts_speed = gr.Slider(label="TTS Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                tts_voice = gr.Dropdown(label="Voice", choices=list(voice_menu.keys()), value="af_alloy (en)")
            tts_audio_output = gr.Audio(label="TTS Audio Output", interactive=False)
    
    run_btn = gr.Button("Run Speak2me Pipeline", variant="primary")
    run_btn.click(fn=speak2me_pipeline,
                  inputs=[audio_input, ollama_url, ollama_model, system_prompt, temperature, top_k, top_p, format_val, debug, tts_speed, tts_voice],
                  outputs=[transcription_display, ollama_response_display, tts_audio_output])
    
demo.launch(server_name="127.0.0.1", share=False)
