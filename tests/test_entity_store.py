import json

from knowledge_graph.entity_store import EntityStore
from knowledge_graph.models import Entity, KnowledgeGraph


def test_entity_store_exists(tmp_path):
    store_path = tmp_path / "kg.json"
    store = EntityStore(path=store_path)

    assert store.exists() is False
    store_path.write_text("{}")
    assert store.exists() is True


def test_entity_store_load_not_exists(tmp_path):
    store_path = tmp_path / "kg.json"
    store = EntityStore(path=store_path)

    graph = store.load()
    assert isinstance(graph, KnowledgeGraph)
    assert len(graph.entities) == 0


def test_entity_store_load_corrupted_json(tmp_path):
    store_path = tmp_path / "kg.json"
    store_path.write_text("{ corrupt json")
    store = EntityStore(path=store_path)

    graph = store.load()
    assert isinstance(graph, KnowledgeGraph)
    assert len(graph.entities) == 0


def test_entity_store_load_valid_json(tmp_path):
    store_path = tmp_path / "kg.json"
    valid_data = {
        "entities": [
            {"name": "AAPL", "entity_type": "COMPANY", "properties": {"fullname": "Apple"}}
        ],
        "relationships": [
            {"source": "AAPL", "target": "TIM_COOK", "relation": "CEO", "properties": {}}
        ],
    }
    store_path.write_text(json.dumps(valid_data))

    store = EntityStore(path=store_path)
    graph = store.load()

    assert isinstance(graph, KnowledgeGraph)
    assert len(graph.entities) == 1
    assert len(graph.relationships) == 1


def test_entity_store_save(tmp_path):
    store_path = tmp_path / "kg.json"
    store = EntityStore(path=store_path)

    graph = KnowledgeGraph()
    graph.add_entity(Entity(name="AAPL", entity_type="COMPANY"))

    store.save(graph)

    assert store_path.exists()
    data = json.loads(store_path.read_text())
    assert len(data["entities"]) == 1
    assert data["entities"][0]["name"] == "aapl"
