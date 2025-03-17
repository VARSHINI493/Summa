
import whisper
import gradio as gr
import datetime
from pydub import AudioSegment

# Load the model once (use "tiny" for faster results)
model = whisper.load_model("tiny")  # Options: "tiny", "small", "base"

def transcribe(inputs, timestamp):
    if not inputs:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    # Convert audio to mono & 16kHz for faster processing
    audio = AudioSegment.from_file(inputs)
    audio = audio.set_channels(1).set_frame_rate(16000)
    temp_audio_path = "processed_audio.wav"
    audio.export(temp_audio_path, format="wav")

    # Transcribe processed audio
    result = model.transcribe(temp_audio_path)

    if timestamp == "Yes":
        output = ""
        if "segments" in result:  
            for segment in result["segments"]:
                start_time = str(datetime.timedelta(seconds=segment["start"]))
                end_time = str(datetime.timedelta(seconds=segment["end"]))
                output += f"{start_time} --> {end_time}\n{segment['text'].strip()}\n\n"
        else:
            output = "No transcription segments found."
    else:
        output = result.get("text", "No transcription available.")

    return output

# Define the Gradio interface
interface = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources=["upload"], type="filepath"),
        gr.Radio(["Yes", "No"], label="Timestamp", info="Displays with timestamp if needed."),
    ],
    outputs="text",
    title="Transcribe Audio",
    description="Audio to Text Convertion."
)

# Launch the interface
interface.launch()
