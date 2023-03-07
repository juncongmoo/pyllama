rm -rf llama_env
python3 -m venv llama_env
source llama_env/bin/activate

pip uninstall llama -U
pip install pyllama -U
pip install gradio

