<p align="center">
<img 
  width="20%"
  src="https://github.com/josejuanmartinez/mindcraft/assets/36634572/0ef83288-9e53-444d-baa0-2c61b0fc26ca" alt="mindcraft"/>
</p>

# MindCraft
The open-source AI solution to craft the minds of your NPC characters for your video games.

It includes the following features:

- Text generation using LLMs (Mistral)
- Motivations, personality, personal goals
- Knowledge and awareness about the world (RAG)
- Short and Long-term memory (RAG)
- Conversational styles
- Supervised finetuning by human feedback

## CUDA and Torch
Although torch is included in the `transformers` library as a dependency, if you see your gpu is not being
utilized, try to run:
- For Cuda 12.1, 12.2, 12.3:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
<p align="center">
<img 
  width="50%"
  src="https://github.com/josejuanmartinez/mindcraft/assets/36634572/7778d4a4-6b25-4b1a-9b26-b1bfa9d94727" alt="mindcraft architecture"/>
</p>
