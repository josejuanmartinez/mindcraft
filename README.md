<p align="center">
<img width="20%" src="https://github.com/josejuanmartinez/mindcraft/assets/36634572/0ef83288-9e53-444d-baa0-2c61b0fc26ca" alt="mindcraft"/>
</p>

# MindCraft
The open-source NLP library to craft the minds of your NPC characters for your video games.

Requires Python 3.10 or higher.

It includes the following features:

- Text generation using LLMs (Mistral)
- Motivations, personality, personal goals
- Knowledge and awareness about the world (RAG)
- Short and Long-term memory (RAG)
- Conversational styles
- Supervised finetuning by human feedback (SFT)
- Integration with vLLM for fast-inference (both local and Docker)
- Usage of quantized AWQ models
- Integration with API and RPC (to come!)

## Create a World from a book
```python
from mindcraft.infra.engine.llm_types import LLMType
from mindcraft.infra.splitters.text_splitters_types import TextSplitterTypes
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.lore.world import World

world = World(world_name="Middle Earth from the Lord of the Rings",
              embeddings=EmbeddingsTypes.MINILM,
              store_type=StoresTypes.CHROMA,
              llm_type=LLMType.ZEPHYR7B_AWQ,
              fast=False)

world.book_to_world(book_path="/content/lotr1.txt",
                    text_splitter=TextSplitterTypes.SENTENCE_SPLITTER,
                    max_units=3,
                    overlap=1,
                    encoding='utf-8')
```
## Query the lore of the world
Once a world has been created and populated with lore, query the lore known by NPCs by doing:
```python
results = world.get_lore("What do you think about the Rings of Power?", num_results=5, min_similarity=0.95)
for i, d in enumerate(results.documents):
    print(f"SENTENCE {i}:\n{d}")
    print()
```

## Instantiate an NPC in a world
Once a world has been created and populated with lore, instantiate an NPC in it by doing:

```python
from mindcraft.features.motivation import Motivation
from mindcraft.features.personality import Personality
from mindcraft.features.mood import Mood

from mindcraft.mind.npc import NPC

name = "Galadriel"
description = "The Elven Queen of Lothlorien, bearer of Nenya, wife to Celeborn"
personalities = [Personality(x) for x in ['fair', 'mighty', 'wise', 'carying', 'kind']]
motivations = [Motivation(x) for x in ['Destroying the Evil', 'Protecting Middle Earth', 'Travelling West']]
mood = Mood('worried')

galadriel = NPC(name,
                description,
                personalities,
                motivations,
                mood,
                StoresTypes.CHROMA,
                EmbeddingsTypes.MINILM)

galadriel.add_npc_to_world()
```

# Ask questions to the NPC
```python
answer, _ = galadriel.react_to("What do you think about the Rings of Power",
                               min_similarity=0.85,
                               ltm_num_results=3,
                               world_num_results=7,
                               max_tokens=600)
```

### Example of answer using Zephyr 7B quantized to 4b
```Alas, my dear friend, the Rings of Power are a perilous burden that should not be wielded by those of mortal race. They possess a sinister force that overcomes the spirit, enslaving their very soul. I have observed their destructive potential in Eregion long ago, where many Elven-rings were created, each infused with its own unique potency. Some more potent than others, and often, they seem to have a propensity to gravitate towards the unsuspecting. In light of our shared concern for the wellbeing of Middle Earth, I implore you to heed my words; let us not succumb to the allure of these fateful rings, lest they consume us entirely.```

## Creating custom Supervised Finetuning datasets for NPCs
There are two loops integrated in the framework which allow you to create your own datasets.
1. `NPC.extract_conversational_styles_from_world`: Retrieving conversations from the world and tagging the question and mood yourself
2. `NPC.extract_conversational_styles_talking_to_user`: Creates conversations by talking to you

### Supervised Finetuning
You can create  your own datasets with supervised finetuning,
accepting or rejecting the interactions. As a result, you will come up
with a csv of interactions you can use to train or finetune your own models.

```csv
name||mood||question||answer
Galadriel||default||Good night, Galadriel!||'Good night, my friends! '\nsaid Galadriel. '\nSleep in peace!
Galadriel||grave||why he could say that?||....`He would be rash indeed that said that thing,' said Galadriel gravely.
```

## LLM integrated (4b quantization with AWQ)
- TheBloke/openinstruct-mistral-7B-AWQ
- TheBloke/zephyr-7B-beta-AWQ
- TheBloke/dragon-yi-6B-v0-AWQ

## Embeddings for RAG
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## CUDA and Torch
Although torch is included in the `transformers` library as a dependency, if you see your gpu is not being
utilized, try to run:
```
pip3 install torch -i https://download.pytorch.org/whl/cu121
```

## Fast Inference
`vLLM` has been included for Fast Inference, in both local installation and installed using Docker.
To use fast-inference, just run add `fast=True` to your `World` object:
```python
world = World(world_name="Lord of the Rings",
              embeddings=EmbeddingsTypes.MINILM,
              store_type=StoresTypes.CHROMA,
              llm_type=LLMType.YI6B,
              fast=True) # <---- HERE
```

## Example data
[Lord of the Rings](https://www.kaggle.com/datasets/ashishsinhaiitr/lord-of-the-rings-text)

## Demo 1: Creating a World and an NPC
[Video](https://youtu.be/T-D1KVIuvjA)

## Architecture
<p align="center">
<img 
  width="50%"
  src="https://github.com/josejuanmartinez/mindcraft/assets/36634572/7778d4a4-6b25-4b1a-9b26-b1bfa9d94727" alt="mindcraft architecture"/>
</p>

## Header
<p align="center">
<img width="100%" src="https://github.com/josejuanmartinez/mindcraft/blob/main/galadriel.png" alt="galadriel"/>
</p>
