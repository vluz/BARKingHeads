import torch
import streamlit as st
import gc, time
from bark import SAMPLE_RATE, generate_audio, preload_models
from IPython.display import Audio
from scipy.io.wavfile import write as write_wav


@st.cache_resource
def load_models():
    preload_models()


load_models()
torch.cuda.empty_cache()
gc.collect()
st.title("BARK TTS test")
st.divider()
st.text("Special signals: ")
st.text("[laughter]  [laughs]  [sighs]  [music]  [gasps]  [clears throat]")
st.text("— or ... for hesitations and ♪ for song lyrics")
st.text("CAPITALIZATION for emphasis of a word")
st.text("MAN/WOMAN: for bias towards a speaker")
st.divider()
text_prompt = st.text_area("Prompt", """[clears throat] HELLO! Uh — I like pizza. [laughs] 
But I have other interests such as games [sighs]""", height=150)
if st.button("Generate"):
    st.divider()
    with st.spinner("Generating..."):
        audio_array = generate_audio(text_prompt)
        filename = time.strftime("%Y%m%d-%H%M%S") + ".wav"
        write_wav(filename, SAMPLE_RATE, audio_array)
        torch.cuda.empty_cache()
        f = open(filename, "rb")
        file = f.read()
        f.close()
    st.write("Result:")
    st.audio(file)
torch.cuda.empty_cache()
gc.collect()
