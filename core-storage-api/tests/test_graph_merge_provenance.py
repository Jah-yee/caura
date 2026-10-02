"""Graph references survive merge and text-mined links reset with their source."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select, update

from common.constants import VECTOR_DIM
from common.models import Entity, Memory, MemoryEntityLink
from common.models.entity import LINK_SOURCE_EXTRACTION
from core_storage_api.services.postgres_service import PostgresService, get_session

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def _memory(svc: PostgresService, tenant: str, content: str, **extra):
    return await svc.memory_add(
        {
            "tenant_id": tenant,
            "agent_id": "graph-tester",
            "content": content,
            "memory_type": "fact",
            "weight": 0.5,
            "status": "active",
            "visibility": "scope_agent",
            **extra,
        }
    )


async def test_duplicate_merge_repoints_memory_subject(_ensure_schema):
    svc = PostgresService()
    tenant = f"graph-merge-{uuid.uuid4().hex}"
    canonical = await svc.entity_add(
        {"tenant_id": tenant, "entity_type": "person", "canonical_name": "Ada Lovelace"}
    )
    dupe = await svc.entity_add(
        {"tenant_id": tenant, "entity_type": "person", "canonical_name": "Ada L."}
    )
    memory = await _memory(svc, tenant, f"Ada wrote notes {uuid.uuid4()}")
    async with get_session() as session:
        await session.execute(
            update(Memory).where(Memory.id == memory.id).values(subject_entity_id=dupe.id)
        )
        canonical_row = await session.get(Entity, canonical.id)
        dupe_row = await session.get(Entity, dupe.id)
        assert canonical_row is not None and dupe_row is not None
        await svc._entity_merge_dupe_into_canonical(session, canonical_row, dupe_row, tenant)

    async with get_session() as session:
        subject = await session.scalar(select(Memory.subject_entity_id).where(Memory.id == memory.id))
        missing_dupe = await session.get(Entity, dupe.id)
    assert subject == canonical.id
    assert missing_dupe is None


async def test_cross_link_is_extraction_owned_and_cleared_on_edit_reset(_ensure_schema):
    svc = PostgresService()
    tenant = f"graph-crosslink-{uuid.uuid4().hex}"
    vector = [1.0] + [0.0] * (VECTOR_DIM - 1)
    memory = await _memory(svc, tenant, f"Alice visited {uuid.uuid4()}", embedding=vector)
    entity = await svc.entity_add(
        {
            "tenant_id": tenant,
            "entity_type": "person",
            "canonical_name": "Alice",
            "name_embedding": vector,
        }
    )
    result = await svc.entity_discover_cross_links(
        tenant_id=tenant,
        fleet_id=None,
        batch_size=10,
        threshold=0.99,
        text_verify=True,
        target_memory_ids=[memory.id],
    )
    assert result["links_created"] == 1
    async with get_session() as session:
        source = await session.scalar(
            select(MemoryEntityLink.source).where(
                MemoryEntityLink.memory_id == memory.id,
                MemoryEntityLink.entity_id == entity.id,
            )
        )
    assert source == LINK_SOURCE_EXTRACTION

    await svc.memory_reset_entity_artifacts(tenant, memory.id)
    async with get_session() as session:
        link = await session.scalar(
            select(MemoryEntityLink).where(
                MemoryEntityLink.memory_id == memory.id,
                MemoryEntityLink.entity_id == entity.id,
            )
        )
    assert link is None
