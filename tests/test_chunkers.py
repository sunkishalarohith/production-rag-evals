from rag.chunkers import fixed_chunks, split_sections, structure_chunks

FILING = """Item 1. Business
Apple Inc. designs and sells consumer electronics.
The company also offers services.
Item 1A. Risk Factors
Supply shortages could harm results.
Competition is intense in every market.
Item 2. Properties
The company leases facilities worldwide.
"""


def test_fixed_chunks_overlap_and_cover():
    chunks = fixed_chunks(FILING, "AAPL", chunk_words=10, overlap=3)
    assert len(chunks) >= 2
    assert all(c.doc_id == "AAPL" for c in chunks)
    # The last word of the document appears in the final chunk.
    assert "worldwide." in chunks[-1].text
    # Indices are sequential.
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_fixed_chunks_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        fixed_chunks(FILING, "AAPL", chunk_words=10, overlap=10)


def test_split_sections_on_item_headings():
    sections = split_sections(FILING)
    headings = [h for h, _ in sections]
    assert "Item 1. Business" in headings
    assert "Item 1A. Risk Factors" in headings
    risk = dict(sections)["Item 1A. Risk Factors"]
    assert "Supply shortages" in risk


def test_structure_chunks_keep_section():
    chunks = structure_chunks(FILING, "AAPL", max_words=500)
    risk_chunks = [c for c in chunks if "Supply shortages" in c.text]
    assert risk_chunks
    assert risk_chunks[0].section == "Item 1A. Risk Factors"
    assert risk_chunks[0].text.startswith("Item 1A. Risk Factors")


def test_structure_chunks_size_cap():
    big = "Item 1. Business\n" + " ".join(["word"] * 1200)
    chunks = structure_chunks(big, "AAPL", max_words=500, overlap=50)
    assert len(chunks) >= 3
    for c in chunks:
        # heading (3 words) + at most max_words of body
        assert len(c.text.split()) <= 503
