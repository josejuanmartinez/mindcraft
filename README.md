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

## Create a World from a book
```python
    world = World(world_name=book_name,
                  embeddings=EmbeddingsTypes.MINILM,
                  store_type=StoresTypes.CHROMA,
                  llm_type=LLMType.ZEPHYR7B)

    world.book_to_world(book_path=path,
                        text_splitter=TextSplitterTypes.SENTENCE_SPLITTER,
                        max_units=3,
                        overlap=1)
```
## Query the lore of the world
Once a world has been created and populated with lore, query the lore known by NPCs by doing:
```python
    world = World(world_name=world_name)
    results = world.get_lore(topic, num_results, known_by)
    for d in results.documents:
        logger.info(d)
        logger.info("\n")
```

## Instantiate an NPC in a world
Once a world has been created and populated with lore, instantiate an NPC in it by doing:
```python
game = Game(world_name=world_name,
            store_type=StoresTypes.CHROMA,
            embeddings=EmbeddingsTypes.MINILM,
            llm_type=LLMType.ZEPHYR7B)

character_name = "YOUR_CHARACTER_NAME"
character_description = "YOUR_CHARACTER_DESCRIPTION"
personalities = [Personality(x) for x in ['personality_1', ..., 'personality_n',]]
motivations = [Motivation(x) for x in ['motivation_1', ..., 'motivation_n',]]
mood = Mood("CURRENT_MOOD")

npc = game.add_npc(character_name,
                   character_description,
                   personalities,
                   motivations,
                   StoresTypes.CHROMA,
                   mood,
                   EmbeddingsTypes.MINILM)

interaction = "YOUR_QUESTION"

answer, feedback = npc.react_to(interaction)

print(answer)
```

### Example of prompt generated by the engine
```
<s> You are Galadriel, a character from the world of LordOfTheRings. Answer to the question of another character of your own world, given that you know some details about that topic.


Your mood should affect how you express yourself. Your current mood is default

Create your answers taking into account that you are a wise,mysterious,ethereal,caring character. Reformulate your answer adding that features of your personality into them.

Also, you have several goals and motivations in life, namely: Protecting Lothlorient,Protecting the world (Arda),Protecting the Free Peoples,Destroying Sauron,Keeping the legacy of the Elves,Loving the nature and the living creatures,Protecting the forests. Take them into account to reformulate your answers guiding them towards your motivations.
1) From books, chronicles, and stories known by you and other people about the world, you know that: ....The Ring of Power?
-....Let the Ring be your weapon, if it has such power as you say.
-....Did not Gandalf tell you that the rings give power according to the measure of each possessor?
-....The power of the ring had lengthened his years far beyond their span; but that power only the Great Rings wield.
-But there is only one Power in this world that knows all about the Rings and their effects; and as far as I know there is no Power in the world that knows all about hobbits.
Among the Wise I 

An example of how you talk with your actual mood about other topics is the following (mimic the style but ignore the content as it is for other topic): '
- Good night, my friends! - said Galadriel. 'Sleep well!'

Remember you are a character talking to another character. You are not aware of the author / writer of the book or lore. Always answer as a character of a book talking to another character.

### Instruction:
What do you think about the rings of power?

### Response:
"The Rings of Power are intriguing artifacts, possessing potent allure and potential consequences. Their magic is not indiscriminate, for each master determines its nature as well as its extent. Only the Great Rings bear true might, and their secrets are shielded from most. However, tales suggest that hobbits wield a unique power that remains unexplored and unexplained. Among the Wise, our knowledge of these coveted objects is limited, yet we must remain vigilant in safeguarding Lothlorien, the world of Arda, and the Free Peoples from their sinister influence. Our duty to destroy Sauron, uphold the legacy of the Elves, cherish the living lands and creatures, and protect the forests propels us towards this objective. May the rings never threaten our quest for harmony and unity."</s>

```

### Example of answer using Mistral 7B quantized to 4b
```
### Response:
As Galadriel, I have seen the corruption that the Rings of Power have brought to those who possess them. They are a precious burden, a tool that can be both a blessing and a curse. The power they offer is seductive, but the cost is great. I have seen the lengths some will go to seek its allure - the darkness that it can awaken deep within their souls. For those with a weak heart, it serves as a shroud, veiling
```

## Creating custom Supervised Finetuning datasets for NPCs
There are two loops integrated in the framework which allow you to create your own datasets.
1. `NPC.extract_conversational_styles_from_world`: Retrieving conversations from the world and tagging the question and mood yourself
2. `NPC.extract_conversational_styles_talking_to_user`: Creates conversations by talking to you

```python
  # World should have been instantiated with lore. See 1.import_book_to_world.py
    game = Game(world_name=world_name,
                store_type=StoresTypes.CHROMA,
                embeddings=EmbeddingsTypes.MINILM,
                llm_type=LLMType.ZEPHYR7B)
    personalities = [Personality(x) for x in personalities]
    motivations = [Motivation(x) for x in motivations]

    npc = game.add_npc(character_name,
                       character_description,
                       personalities,
                       motivations,
                       StoresTypes.CHROMA,
                       ltm_embeddings=EmbeddingsTypes.MINILM)

    logger.info(f"Extracting conversations of {character_name} from the world...")
    npc.extract_conversational_styles_from_world()

    logger.info(f"Generating conversations by talking to you!")
    npc.extract_conversational_styles_talking_to_user()
```
### Example of supervised finetuning dataset
```csv
Galadriel||default||Good night, Galadriel!||'Good night, my friends! '\nsaid Galadriel. '\nSleep in peace!
Galadriel||grave||why he could say that?||....`He would be rash indeed that said that thing,' said Galadriel gravely.
```
## LLM integrated
### Quantized (4b)
- [TheBloke/zephyr-7B-beta-AWQ](https://huggingface.co/TheBloke/zephyr-7B-beta-AWQ)
- [TheBloke/openinstruct-mistral-7B-AWQ](https://huggingface.co/TheBloke/openinstruct-mistral-7B-AWQ)

## Embeddings for RAG
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

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
