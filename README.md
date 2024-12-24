# NIM and Pipecat AI Jupyter Notebook

A blueprint notebook showcasing Pipecat AI and NIM in the creation of an AI voice bot. It uses the `nvidia/llama-3.1-nemotron-70b-instruct` LLM model and Riva for STT & TTS. The intention is for this to become a launchable on the brev.dev platform.

Pipecat AI is an open-source framework for building voice and multimodal conversational agents. Pipecat simplifies the complex voice-to-voice AI pipeline, and lets developers build AI capabilities easily and with Open Source, commercial, and custom models. The framework was developed by [Daily](https://daily.co/), a company that has provided real-time video and audio communication infrastructure since 2016. It is fully vendor neutral and is not tightly coupled to Daily's infrastructure.

## Setup JupyterLab
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=venv --display-name "Python3.12"
```

## Setup environment
Add NVIDIA API key to `.env`.
```bash
cp example.env .env
# ... edit .env
```

## Run the Jupyter Notebook
```bash
python -m jupyter notebook
```

Navigate to [`http://localhost:8888/notebooks/001-hello-pipecat-nim.ipynb`](http://localhost:8888/notebooks/001-hello-pipecat-nim.ipynb)

## Extras

### Run in a command line environment
For convenience, a standalone pipecat can be found [here](./001-hello-pipecat-nim.py)

```bash
source .env
python3.12 -m venv venv
source venv/bin/activate
pip install "git+https://github.com/pipecat-ai/pipecat.git@main"
pip install "pipecat-ai[daily,openai,riva,silero]"
pip install noaa_sdk
python 001-hello-pipecat-nim.py
```

### Pipecat-AI links and resources

- [Pipecat Repo](https://github.com/pipecat-ai/pipecat)
- [Pipecat docs](https://docs.pipecat.ai)
