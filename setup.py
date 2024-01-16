from setuptools import find_packages, setup

setup(
    name="mindcraft",
    version="0.2.3",
    packages=find_packages(),
    package_data={p: ["*"] for p in find_packages()},
    url="https://github.com/josejuanmartinez/mindcraft",
    download_url="https://github.com/josejuanmartinez/mindcraft",
    license="MIT",
    install_requires=[
        "transformers==4.35.2",
        "sentence_transformers==2.2.2",
        "autoawq==0.1.7",
        "requests~=2.31.0",
        "torch==2.1.1"
    ],
    python_requires=">=3.10.0",
    author="Juan.Martinez",
    author_email="jjmcarrascosa@gmail.com",
    description="Mindcraft: an LLM-based engine for creating real NPCs. Empowered by Hugging Face, quantized LLMs with "
                "AWQ (thanks @TheBloke) and vLLM. It follows a RAG approach with chunk or sentence splitting, and a "
                "vector store. Right now, ChromaDB is the supported Vector Store and chunk splitting using `tiktoken` "
                "or sentence splitting using `spacy` are available.",
    keywords=["nlp", "llm", "npc", "conversations", "videogames"],
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.1',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Games/Entertainment'
      ]
)
