import os
import tempfile
import unittest

from mindcraft.features.mood import Mood
from mindcraft.features.motivation import Motivation
from mindcraft.features.personality import Personality
from mindcraft.mind.npc import NPC
from mindcraft.infra.engine.llm_types import LLMType
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.lore.world import World
import pytest


@pytest.fixture
def temp_file():
    # Create a temporary file
    temp_fd, temp_path = tempfile.mkstemp()

    # Close the file descriptor, as we don't need it
    os.close(temp_fd)

    # Provide the temporary file path to the test
    yield temp_path

    # Clean up the temporary file after the test
    os.remove(temp_path)


def test_stm_retrieves_last(tmp_path):
    world = World(world_name="TheAgeOfSigmur",
                  embeddings=EmbeddingsTypes.MINILM,
                  store_type=StoresTypes.CHROMA,
                  llm_type=LLMType.ZEPHYR7B_AWQ,
                  path=tmp_path)

    personalities = [Personality(x) for x in ['chaotic']]
    motivations = [Motivation(x) for x in ['Kill everyone']]
    mood = Mood('angry')

    npc = NPC("Sigmur",
              "An evil orc seeking destruction",
              personalities,
              motivations,
              mood,
              StoresTypes.CHROMA,
              EmbeddingsTypes.MINILM)
    npc.add_npc_to_world()
    npc._ltm.memorize(
        "In the rugged, untamed lands of the fantasy realm, Orcs stand as imposing figures, embodying strength, ferocity, and a primal connection to the wilderness. Towering in stature, an average Orc stands between 6 to 7 feet tall, their robust, muscular frames attesting to their formidable physical prowess. Characterized by coarse, green or grayish skin, and prominent tusks protruding from their lower jaws, Orcs present an intimidating visage that strikes fear into the hearts of those who cross their path.",
        mood)
    npc._ltm.memorize(
        "Orcish society is tribal and closely tied to the natural world. Living in tight-knit clans, they roam vast territories, mastering the art of survival in harsh environments. Orcs are skilled hunters and gatherers, relying on their keen senses and deep understanding of the land to sustain their communities. Despite their fierce reputation, Orcs value loyalty and honor within their tribes, fostering a strong sense of camaraderie among their kind.",
        mood)
    npc._ltm.memorize(
        "While Orcs are often misunderstood and perceived as savages by other races, they possess a rich cultural heritage. Tribal shamans play a crucial role in preserving their history and traditions, passing down tales of ancient battles and heroic deeds through oral storytelling. These sagas often center around legendary Orcish warriors who have left an indelible mark on their people, inspiring the younger generation to aspire to greatness.",
        mood)
    npc._ltm.memorize(
        "In times of conflict, Orcs are formidable adversaries on the battlefield, wielding brutal weaponry and displaying a remarkable resilience to pain. Despite their martial prowess, many Orcs seek a balance between the chaos of war and the tranquility of the natural world. Some individuals, known as Orcish druids, harness the primal forces of nature, further connecting their people to the land and its energies.",
        mood)
    npc._ltm.memorize(
        "While Orcs may be portrayed as antagonists in many fantasy tales, a closer look reveals a multifaceted race with a complex blend of strength, honor, and a deep-rooted connection to the wilderness. Whether as fearsome warriors defending their homelands or as guardians of ancient traditions, Orcs contribute a unique and captivating element to the diverse tapestry of fantasy worlds.",
        mood)

    interaction = "The elves are an aberration!"
    summary = npc._stm.refresh_summary(last_interaction=interaction)
    assert len(npc._stm.summary) != 0
    assert interaction in summary


if __name__ == '__main__':
    unittest.main()
