from langchain_text_splitters import RecursiveCharacterTextSplitter


def test_text_splitter_logic():
    text = "A" * 2500

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([text])

    # 2500 characters should produce multiple chunks
    assert len(chunks) > 1

    # No chunk should be larger than our configured chunk size
    for chunk in chunks:
        assert len(chunk.page_content) <= 1000


def test_empty_documents():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([])

    # No input documents should produce no chunks
    assert chunks == []