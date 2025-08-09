from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import JSON

from src.core import models

# Monkeypatch JSONB to JSON for SQLite
models.JSONB = JSON
Base = models.Base
Agent = models.Agent
Task = models.Task


def patch_jsonb_to_json():
    from sqlalchemy.types import JSON

    for cls in Base.__subclasses__():
        for col in cls.__table__.columns:
            if col.type.__class__.__name__ == "JSONB":
                col.type = JSON()


def setup_in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    patch_jsonb_to_json()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_agent_instantiation_and_short_id():
    session = setup_in_memory_db()
    agent = Agent(name="test_agent", type="meta", role="tester", status="active")
    session.add(agent)
    session.commit()
    assert agent.id is not None
    assert agent.short_id is not None
    assert agent.name == "test_agent"
    assert agent.status == "active"
    # Relationship: tasks
    assert agent.tasks == []
    session.close()


def test_task_instantiation_and_relationship():
    session = setup_in_memory_db()
    agent = Agent(name="test_agent2", type="meta", role="tester", status="active")
    session.add(agent)
    session.commit()
    task = Task(title="Test Task", type="unit", agent_id=agent.id)
    session.add(task)
    session.commit()
    assert task.id is not None
    assert task.short_id is not None
    assert task.agent_id == agent.id
    # Relationship: agent.tasks
    assert task in agent.tasks
    session.close()


def test_event_listener_short_id_generation():
    session = setup_in_memory_db()
    agent = Agent(name="listener_agent", type="meta", role="tester", status="active")
    session.add(agent)
    session.commit()
    assert agent.short_id[:3] == "lis"
    assert len(agent.short_id) == 5
    assert agent.short_id.isalnum()
    task = Task(title="Listener Task", type="unit", agent_id=agent.id)
    session.add(task)
    session.commit()
    assert len(task.short_id) == 4
    assert task.short_id.isdigit()
    session.close()
