"""Permanent tenant-isolation guard (R5/N1) — the only committed test.

Uploads a same-named document for two tenants through the real HTTP API
and asserts that every response scoped to tenant A carries zero tenant-B
content: search results, document listings, and single-document reads.
Cross-tenant deletes are proven to be no-ops. Any future endpoint or
query change that breaks tenant isolation must fail here.

Requires the Docker Qdrant instance and a configured ``OPENAI_API_KEY``
(real embeddings — this guard never stubs the provider). Skips with a
clear message when either is unavailable.
"""

import uuid

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.infrastructure.settings import Settings
from app.main import app


def _skip_reason() -> str:
    try:
        settings = Settings()
    except Exception as error:
        return f"settings unavailable ({type(error).__name__}: {error})"
    try:
        reachable = (
            httpx.get(settings.QDRANT_URL, timeout=2.0).status_code == 200
        )
    except httpx.HTTPError:
        reachable = False
    if not reachable:
        return f"Qdrant not reachable at {settings.QDRANT_URL} — start it with `docker compose up -d`"
    key = settings.OPENAI_API_KEY
    if not key.startswith("sk-") or len(key) < 40 or "replace" in key.lower():
        return "OPENAI_API_KEY not configured (placeholder or missing in .env) — set a real key to run the isolation guard"
    return ""


_REQUIRES_INFRASTRUCTURE = _skip_reason()
pytestmark = pytest.mark.skipif(
    bool(_REQUIRES_INFRASTRUCTURE), reason=_REQUIRES_INFRASTRUCTURE or "skipped"
)


@pytest.fixture(scope="module")
def client():
    """Context-managed TestClient so the app's startup bootstrap runs
    (in-process, no server needed)."""
    with TestClient(app) as test_client:
        yield test_client


def _pdf_with_text(text: str) -> bytes:
    """Build a small digital PDF whose page text is exactly ``text``."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    data = document.tobytes()
    document.close()
    return data


def _upload(client: TestClient, tenant: str, filename: str, content: bytes) -> dict:
    response = client.post(
        f"/tenants/{tenant}/documents",
        files={"file": (filename, content, "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ready", body
    assert body["num_chunks"] > 0, body
    return body


def _search(client: TestClient, tenant: str, query: str) -> list[dict]:
    response = client.post(f"/tenants/{tenant}/search", json={"query": query, "top_k": 10})
    assert response.status_code == 200, response.text
    return response.json()


def test_tenant_isolation(client: TestClient) -> None:
    run = uuid.uuid4().hex[:8]
    tenant_a = f"iso-a-{run}"
    tenant_b = f"iso-b-{run}"
    marker_a = f"ALPHA{run}ZULU"
    marker_b = f"BRAVO{run}YANKEE"
    filename = "isolation-policy.pdf"  # same name in both tenants

    pdf_a = _pdf_with_text(f"{marker_a} refund policy: 30 day money-back guarantee.")
    pdf_b = _pdf_with_text(f"{marker_b} refund policy: store credit only.")

    try:
        doc_a = _upload(client, tenant_a, filename, pdf_a)["document_id"]
        doc_b = _upload(client, tenant_b, filename, pdf_b)["document_id"]
        assert doc_a != doc_b

        # Search as A: hits exist, every hit is A's document, and zero
        # tenant-B content ever appears (R5/N1).
        hits_a = _search(client, tenant_a, f"refund policy {marker_a}")
        assert hits_a, "expected hits for tenant A's own document"
        assert all(hit["document_id"] == doc_a for hit in hits_a)
        assert all(marker_b not in hit["content"] for hit in hits_a)
        assert any(marker_a in hit["content"] for hit in hits_a)

        # Symmetric from B — isolation is bidirectional.
        hits_b = _search(client, tenant_b, f"refund policy {marker_b}")
        assert hits_b, "expected hits for tenant B's own document"
        assert all(hit["document_id"] == doc_b for hit in hits_b)
        assert all(marker_a not in hit["content"] for hit in hits_b)

        # Listing as A mentions only A's document — never B's.
        listing_a = client.get(f"/tenants/{tenant_a}/documents")
        assert listing_a.status_code == 200, listing_a.text
        assert [doc["document_id"] for doc in listing_a.json()] == [doc_a]

        # A wrong-tenant GET is indistinguishable from a wrong-id GET —
        # no tenant-existence oracle.
        cross = client.get(f"/tenants/{tenant_a}/documents/{doc_b}")
        wrong = client.get(f"/tenants/{tenant_a}/documents/{uuid.uuid4()}")
        assert cross.status_code == 404 and wrong.status_code == 404
        assert cross.json() == wrong.json()

        # A cross-tenant DELETE is a no-op: B's content stays searchable.
        assert (
            client.delete(f"/tenants/{tenant_a}/documents/{doc_b}").status_code == 204
        )
        hits_b_after = _search(client, tenant_b, f"refund policy {marker_b}")
        assert hits_b_after, "tenant B's document must survive A's cross-tenant delete"
        assert all(hit["document_id"] == doc_b for hit in hits_b_after)
    finally:
        # Clean both tenants so reruns stay green even on failure.
        for tenant in (tenant_a, tenant_b):
            listing = client.get(f"/tenants/{tenant}/documents")
            if listing.status_code == 200:
                for doc in listing.json():
                    client.delete(f"/tenants/{tenant}/documents/{doc['document_id']}")
