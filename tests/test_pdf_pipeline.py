import json
import math
import os
import re
import subprocess
import sys
import pytest
import pymupdf
from app.pipeline import process_pdf, vector_store_payload, ChunkConfig, EmbeddingService
from app.pipeline.ingest import ingest_pdf, IngestionError
from app.pipeline.chunk import detect_sections, chunk_sections, token_count
from app.pipeline.embed import EmbeddingError

class Tokenizer:
    model_max_length = 64
    def encode(self, text, add_special_tokens=True, truncation=False):
        return list(text) + ([0, 0] if add_special_tokens else [])

class Backend:
    tokenizer = Tokenizer()
    max_seq_length = 64
    def encode(self, texts, **kwargs):
        self.seen = texts
        result = []
        for text in texts:
            vector = [0.] * 384
            vector[sum(map(ord, text)) % 384] = 1.
            result.append(vector)
        return result

@pytest.fixture
def service():
    return EmbeddingService(model=Backend())

@pytest.fixture
def pdf(tmp_path):
    def make(pages, *, encryption=False):
        path = tmp_path / 'paper.pdf'
        with pymupdf.open() as doc:
            doc.set_metadata({'title': 'Synthetic science'})
            for lines in pages:
                page = doc.new_page(width=600, height=800)
                for x, y, text, size in lines:
                    page.insert_text((x, y), text, fontsize=size)
            kwargs = {'encryption': pymupdf.PDF_ENCRYPT_AES_256, 'user_pw': 'secret', 'owner_pw': 'owner'} if encryption else {}
            doc.save(path, **kwargs)
        return path
    return make

CFG = ChunkConfig(36, 5, False)

def test_sections_provenance_determinism_handoff(pdf, service):
    path = pdf([[(50, 100, 'Preamble text', 10), (50, 150, '1 Introduction', 16), (50, 180, 'First paragraph.', 10)],
                [(50, 100, 'Continuation text.', 10), (50, 180, '3 Methods', 16), (50, 210, '3.2 Experimental Setup', 14), (50, 240, 'Setup content.', 10)]])
    sections = detect_sections(ingest_pdf(path))
    assert [s.title for s in sections] == ['Preamble', '1 Introduction', '3 Methods', '3.2 Experimental Setup']
    assert sections[-1].path == ['3 Methods', '3.2 Experimental Setup']
    result = process_pdf(path, service=service, config=CFG)
    assert result == process_pdf(path.read_bytes(), source_filename=path.name, service=service, config=CFG)
    assert set(p for c in result['chunks'] if c['section_title'] == '1 Introduction' for p in c['page_numbers']) == {1, 2}
    assert result['metadata']['title'] == 'Synthetic science'
    assert json.loads(json.dumps(result)) == result
    payload = vector_store_payload(result)
    assert len({len(a) for a in payload.values()}) == 1
    for i, c in enumerate(result['chunks']):
        assert payload['ids'][i] == c['chunk_id']
        assert payload['documents'][i] == c['text']
        assert payload['embeddings'][i] == c['embedding']
        assert json.loads(payload['metadatas'][i]['page_numbers']) == c['page_numbers']

def test_single_column_order(pdf):
    texts = [b.text for b in ingest_pdf(pdf([[(50, 200, 'Second', 10), (50, 100, 'First', 10)]])).blocks]
    assert texts == ['First', 'Second']

def test_two_columns_spanning_separator(pdf):
    path = pdf([[(50, 70, 'Full width title with enough text to span the page centre', 16),
                 (50, 120, 'Left first', 10), (340, 120, 'Right first', 10),
                 (50, 180, 'Left second', 10), (340, 180, 'Right second', 10),
                 (50, 260, 'Another full width line separating the next column band', 16),
                 (50, 320, 'Left third', 10), (340, 320, 'Right third', 10)]])
    text = '\n'.join(b.text for b in ingest_pdf(path).blocks)
    assert text.index('Left second') < text.index('Right first')
    assert text.index('Right second') < text.index('Left third')
    assert text.index('Left third') < text.index('Right third')

def test_misleading_and_absent(pdf):
    path = pdf([[(50, 100, 'The Introduction describes our Results.', 10), (50, 140, '3.2 is a measured value.', 10)]])
    assert [s.title for s in detect_sections(ingest_pdf(path))] == ['Document']

@pytest.mark.parametrize('heading', ['Abstract', 'Introduction', 'Background', 'Related Work', 'Methods', 'Results', 'Discussion', 'Conclusion', 'References', 'Appendix'])
def test_unnumbered(pdf, heading):
    assert detect_sections(ingest_pdf(pdf([[(50, 100, heading, 14), (50, 140, 'Body.', 10)]])))[0].title == heading

def test_margin_conservative(pdf):
    paper = ingest_pdf(pdf([[(50, 30, 'Running header', 10), (50, 150, 'Repeated body text', 10), (50, 780, str(i+1), 10)] for i in range(3)]))
    assert all(b.text == 'Repeated body text' for b in paper.blocks)
    assert len(paper.warnings) == 6
    paper = ingest_pdf(pdf([[(50, 30, 'Unique top content', 10), (50, 100, 'Body.', 10)]]))
    assert any(b.text == 'Unique top content' for b in paper.blocks)

@pytest.mark.parametrize('kind', ['missing', 'malformed', 'encrypted', 'image_only'])
def test_errors(pdf, tmp_path, kind):
    if kind == 'missing':
        with pytest.raises(FileNotFoundError):
            ingest_pdf(tmp_path / 'missing.pdf')
    elif kind == 'malformed':
        with pytest.raises(IngestionError, match='Malformed'):
            ingest_pdf(b'not a PDF')
    else:
        path = pdf([[]] if kind == 'image_only' else [[(50, 100, 'Secret', 10)]], encryption=kind == 'encrypted')
        with pytest.raises(IngestionError, match='OCR|Password'):
            ingest_pdf(path)

def test_mixed(pdf):
    paper = ingest_pdf(pdf([[], [(50, 100, 'Usable.', 10)]]))
    assert paper.blocks[0].page == 2 and 'Page 1' in paper.warnings[0]

def test_coverage_boundaries_limits_overlap(pdf, service):
    path = pdf([[(50, 100, 'Introduction', 14), (50, 140, 'First sentence. Second sentence.', 10)] +
                [(50, 180+i*20, 'abcdefghij'*8, 8) for i in range(3)] +
                [(50, 300, 'Results', 14), (50, 340, 'Final body.', 10)]])
    paper = ingest_pdf(path)
    sections = detect_sections(paper)
    chunks = chunk_sections(paper, sections, service.tokenizer, 64, CFG)
    assert len(chunks) < 100
    plain = chunk_sections(paper, sections, service.tokenizer, 64, ChunkConfig(36, 0, False))
    for section in sections:
        source = ''.join(c.text for c in plain if c.section_title == section.title)
        assert re.sub(r'\s+', '', source) == ''.join(re.sub(r'\s+', '', b.text) for b in section.blocks)
        selected = [c for c in chunks if c.section_title == section.title]
        previous_start, previous_end = -1, 0
        coverage = set()
        for c in selected:
            start = c.section_char_start
            assert source[start:c.section_char_end] == c.text
            assert start >= 0 and start <= previous_end
            if previous_start >= 0:
                assert start <= previous_end  # Complete-sentence overlap can be zero.
            coverage.update(range(start, start + len(c.text)))
            previous_start, previous_end = start, start + len(c.text)
        assert coverage == set(range(len(source)))
    assert all(c.token_count <= 64 for c in chunks)
    assert chunks == chunk_sections(paper, sections, service.tokenizer, 64, CFG)
    assert [c.chunk_id for c in chunks] != [c.chunk_id for c in plain]
    assert all(c.section_title in ['Introduction', 'Results'] for c in chunks)

def test_exclusion_prefix_config(pdf, service):
    paper = ingest_pdf(pdf([[(50, 100, 'Introduction', 14), (50, 140, 'Main text.', 10), (50, 180, 'References', 14),
                            (50, 220, 'Citation.', 10), (50, 260, 'Appendix', 14), (50, 300, 'Extra.', 10)]]))
    sections = detect_sections(paper)
    chunks = chunk_sections(paper, sections, service.tokenizer, 64, ChunkConfig(36, 5, True, True, True))
    assert {c.section_title for c in chunks} == {'Introduction'}
    assert chunks[0].embedding_input.startswith('Introduction\n\n')
    assert chunks[0].token_count == token_count(service.tokenizer, chunks[0].embedding_input)
    with pytest.raises(ValueError):
        chunk_sections(paper, sections, service.tokenizer, 64, ChunkConfig(5, 5))
    with pytest.raises(ValueError, match='Heading prefix'):
        chunk_sections(paper, sections, service.tokenizer, 3)

def test_empty_lazy_reuse_failure_query_order(service):
    calls = []
    def fail():
        calls.append(1)
        raise OSError('download failed')
    failing = EmbeddingService(loader=fail)
    assert failing.embed([]) == [] and not calls
    with pytest.raises(EmbeddingError, match='Unable to load'):
        failing.embed(['x'])
    vectors = service.embed(['alpha', 'beta'])
    assert service.model.seen == ['alpha', 'beta'] and vectors[0] != vectors[1]
    assert service.embed_query('alpha') == vectors[0]
    assert all(len(v) == 384 and math.isclose(sum(x*x for x in v), 1) for v in vectors)
    with pytest.raises(ValueError):
        service.embed_query(' ')
    with pytest.raises(EmbeddingError, match='token limit'):
        service.embed(['x'*64])
    calls.clear()
    reused = EmbeddingService(loader=lambda: calls.append(1) or Backend())
    reused.embed(['x']); reused.embed(['y'])
    assert len(calls) == 1

@pytest.mark.parametrize('vector', [[1.], [float('nan')]*384, [0.]*384, [2.]*384])
def test_invalid_vectors(vector):
    class Bad(Backend):
        def encode(self, texts, **kwargs):
            return [vector for _ in texts]
    with pytest.raises(EmbeddingError):
        EmbeddingService(model=Bad()).embed(['x'])

def test_wrong_count():
    class Bad(Backend):
        def encode(self, texts, **kwargs):
            return []
    with pytest.raises(EmbeddingError, match='number'):
        EmbeddingService(model=Bad()).embed(['x'])

@pytest.mark.skipif(os.environ.get('RUN_MODEL_INTEGRATION') != '1', reason='Opt-in real model download')
def test_real_model_integration(pdf):
    service = EmbeddingService()
    path = pdf([[(50, 80, 'Abstract', 16)] + [(50, 120+i*20, 'Scientific measurements show reproducible experimental results.', 10) for i in range(25)]])
    result = process_pdf(path, service=service)
    assert len(result['chunks']) > 1
    for c in result['chunks']:
        assert c['token_count'] == token_count(service.tokenizer, c['embedding_input']) <= service.model.max_seq_length
        assert len(c['embedding']) == 384 and all(math.isfinite(x) for x in c['embedding'])
        assert math.isclose(sum(x*x for x in c['embedding']), 1, abs_tol=1e-5)
    assert len(service.embed_query('What are the results?')) == 384
    json.dumps(result, allow_nan=False)
    output = path.with_suffix('.json')
    completed = subprocess.run([sys.executable, '-m', 'app.pipeline', str(path), '--output', str(output)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    cli_result = json.loads(output.read_text(encoding='utf-8'))
    assert [c['chunk_id'] for c in cli_result['chunks']] == [c['chunk_id'] for c in result['chunks']]
    assert cli_result['model']['revision']

@pytest.mark.parametrize('heading', ['Introduction', '1 Introduction', '1. Introduction', 'I. INTRODUCTION', 'IV. METHODS'])
def test_heading_number_forms(pdf, heading):
    paper = ingest_pdf(pdf([[(50, 100, heading, 14), (50, 140, 'Body text.', 10)]]))
    assert detect_sections(paper)[0].title == heading


def test_roman_dropcap_and_letter_hierarchy(pdf):
    paper = ingest_pdf(pdf([[(50, 100, 'I. INTRODUCTION', 12), (50, 150, 'D', 32),
                            (80, 150, 'EEP learning provides tools.', 10),
                            (50, 220, 'A. Historical Perspective', 14), (50, 260, 'Historical body.', 10),
                            (50, 300, 'II. METHODS', 12), (50, 340, 'Method body.', 10),
                            (50, 380, '2.1 Experimental Setup', 14), (50, 420, 'Setup body.', 10)]]))
    sections = detect_sections(paper)
    assert [s.title for s in sections] == ['I. INTRODUCTION', 'A. Historical Perspective', 'II. METHODS', '2.1 Experimental Setup']
    assert sections[1].path == ['I. INTRODUCTION', 'A. Historical Perspective']
    assert sections[-1].path == ['II. METHODS', '2.1 Experimental Setup']
    assert re.search(r'D\s*EEP', ' '.join(b.text for b in sections[0].blocks))


@pytest.mark.parametrize('delimiter', [':', '\u2014', '\u2013', '-'])
def test_inline_abstract_unicode(tmp_path, delimiter):
    # Built-in CJK font supports Unicode dash glyphs without system font dependencies.
    path = tmp_path / 'inline.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 100), 'Abstract' + delimiter + 'This paper tests a method.', fontname='china-s', fontsize=10)
        page.insert_text((50, 150), 'Index Terms: imaging, science.', fontname='china-s', fontsize=10)
        page.insert_text((50, 200), 'Introduction', fontsize=14)
        page.insert_text((50, 240), 'Main body.', fontsize=10)
        doc.save(path)
    sections = detect_sections(ingest_pdf(path))
    assert len(sections) == 3
    assert sections[0].title.startswith('Abstract')
    assert sections[0].blocks[0].text == 'This paper tests a method.'
    assert sections[1].title.startswith('Index Terms')
    assert sections[1].blocks[0].text == 'imaging, science.'


def test_standalone_abstract_keywords_and_no_inference(pdf):
    sections = detect_sections(ingest_pdf(pdf([[(50, 100, 'Abstract', 14), (50, 140, 'Summary body.', 10),
                                              (50, 180, 'Keywords', 14), (50, 220, 'Science, imaging.', 10),
                                              (50, 260, 'Introduction', 14), (50, 300, 'Main body.', 10)]])))
    assert [s.title for s in sections] == ['Abstract', 'Keywords', 'Introduction']
    paper = ingest_pdf(pdf([[(50, 100, 'This paper proposes a new method.', 10), (50, 140, 'I. INTRODUCTION', 14), (50, 180, 'Body.', 10)]]))
    assert [s.title for s in detect_sections(paper)] == ['Preamble', 'I. INTRODUCTION']


def test_caption_letter_prose_and_ambiguous_title(pdf):
    paper = ingest_pdf(pdf([[(50, 100, 'D', 30), (50, 160, 'Fig. 1 Results', 16),
                            (50, 220, 'Table 1 Methods', 16), (50, 280, 'Introduction describes our work', 10),
                            (50, 340, '1 Uncertain Title', 10), (50, 400, 'Body text.', 10)]]))
    sections = detect_sections(paper)
    assert [s.title for s in sections] == ['Document']
    assert len(sections[0].blocks) == 6
    assert any('Uncertain Title' in w for w in paper.processing_warnings)


def test_two_column_roman_transition_continuity(pdf):
    paper = ingest_pdf(pdf([[(50, 100, 'I. INTRODUCTION', 14), (50, 140, 'Left content.', 10),
                            (340, 100, 'Right continuation.', 10), (340, 200, 'II. METHODS', 14),
                            (340, 240, 'Method start.', 10)], [(50, 100, 'Method continuation.', 10),
                                                             (50, 200, 'III. RESULTS', 14), (50, 240, 'Result body.', 10)]]))
    sections = detect_sections(paper)
    assert [s.title for s in sections] == ['I. INTRODUCTION', 'II. METHODS', 'III. RESULTS']
    assert [b.text for b in sections[0].blocks] == ['Left content.', 'Right continuation.']
    assert [b.page for b in sections[1].blocks] == [1, 2]
    chunks = chunk_sections(paper, sections, Tokenizer(), 120, ChunkConfig(90, 0, False))
    method = next(c for c in chunks if c.section_title == 'II. METHODS')
    assert method.page_numbers == [1, 2]
    assert all('Result body.' not in c.text for c in chunks if c.section_title != 'III. RESULTS')


def test_complete_sentences_soft_target_and_overlap(pdf):
    paper = ingest_pdf(pdf([[(50, 100, 'Introduction', 14),
                            (50, 140, 'First sentence. Second sentence. Third sentence. Fourth sentence.', 10)]]))
    sections = detect_sections(paper)
    chunks = chunk_sections(paper, sections, Tokenizer(), 120, ChunkConfig(35, 18, False))
    assert len(chunks) >= 2
    assert all(c.text.rstrip().endswith('.') and not c.warnings for c in chunks)
    assert all(c.text.lstrip().split()[0] in {'First', 'Second', 'Third', 'Fourth'} for c in chunks)
    assert any(a.section_char_end > b.section_char_start for a, b in zip(chunks, chunks[1:]))
    # A sentence larger than the target still fits intact under the hard model limit.
    sentence = 'This complete sentence has more than thirty characters.'
    paper = ingest_pdf(pdf([[(50, 100, 'Introduction', 14), (50, 140, sentence, 10)]]))
    chunks = chunk_sections(paper, detect_sections(paper), Tokenizer(), 120, ChunkConfig(30, 5, False))
    assert len(chunks) == 1 and chunks[0].text == sentence and not chunks[0].warnings


def test_exceptional_sentence_report_coverage_and_limits(pdf):
    text = 'abcdefghij' * 6 + '.'
    paper = ingest_pdf(pdf([[(50, 100, 'Methods', 14), (50, 140, text, 8)]]))
    sections = detect_sections(paper)
    config = ChunkConfig(25, 5, True)
    chunks = chunk_sections(paper, sections, Tokenizer(), 40, config)
    assert len(chunks) > 1
    assert all(c.warnings and c.token_count <= 40 for c in chunks)
    assert ''.join(c.text for c in chunks) == text
    assert any('Oversized sentence split' in w for w in paper.processing_warnings)
    assert chunks == chunk_sections(paper, sections, Tokenizer(), 40, config)
    assert all(b.section_char_start > a.section_char_start for a, b in zip(chunks, chunks[1:]))


def test_alternating_small_headers_and_merged_page_numbers(pdf):
    pages = [[(50, 30, 'Alternating title' if i % 2 else 'Alternating journal', 7),
              (540, 30, str(100+i), 7), (50, 150, 'Legitimate repeated body.', 10)] for i in range(7)]
    paper = ingest_pdf(pdf(pages))
    assert all(b.text == 'Legitimate repeated body.' for b in paper.blocks)
    assert len(paper.warnings) == 14


def test_unknown_unnumbered_styled_heading(tmp_path):
    path = tmp_path / 'styled.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 100), 'Introduction', fontsize=14)
        page.insert_text((50, 140), 'Intro body.', fontsize=10)
        page.insert_text((50, 180), 'Experimental Setup', fontname='hebo', fontsize=14)
        page.insert_text((50, 220), 'Setup body.', fontsize=10)
        doc.save(path)
    assert [s.title for s in detect_sections(ingest_pdf(path))] == ['Introduction', 'Experimental Setup']


def test_paragraph_end_preferred_over_partial_next_paragraph(pdf):
    paper = ingest_pdf(pdf([[(50, 100, 'Introduction', 14),
                            (50, 140, 'First sentence. Second sentence.', 10),
                            (50, 180, 'Third sentence. Fourth sentence. Fifth sentence.', 10)]]))
    chunks = chunk_sections(paper, detect_sections(paper), Tokenizer(), 120, ChunkConfig(50, 0, False))
    assert chunks[0].text == 'First sentence. Second sentence.\n\n'
    assert not any(c.warnings for c in chunks)

@pytest.fixture
def numeric_table_pdf(tmp_path):
    def make(long_cell=False, continuation=False):
        path=tmp_path / 'table.pdf'
        with pymupdf.open() as doc:
            for number in range(2 if continuation else 1):
                page=doc.new_page(width=600,height=800)
                page.insert_text((50,90),'Methods',fontsize=14)
                page.insert_text((50,130),'Table 1. Cont.' if number else 'Table 1. Measurements and units.',fontsize=10)
                for y in [150,180,220,320]:
                    page.draw_line((40,y),(560,y))
                for x,t in [(70,'Parameter'),(220,'Value'),(390,'Meaning (MPa)')]: page.insert_text((x,170),t,fontsize=10)
                for x,t in [(70,'Ratio'),(220,'0.48%'),(390,'Percentage')]: page.insert_text((x,205),t,fontsize=10)
                for x,t in [(70,'Strength'),(220,'460 MPa'),(390,'Measured stress')]: page.insert_text((x,240),t,fontsize=10)
                if long_cell:
                    for i in range(5): page.insert_text((380,255+i*11),'Long descriptive cell text',fontsize=8)
                page.insert_text((50,360),'Results',fontsize=14)
                page.insert_text((50,400),'Body results.',fontsize=10)
            doc.save(path)
        return path
    return make

class WideTokenizer(Tokenizer):
    model_max_length=256


def test_table_rows_headers_numeric_context_and_coverage(numeric_table_pdf):
    path=numeric_table_pdf()
    paper=ingest_pdf(path)
    rows=[b for b in paper.blocks if b.kind=='table_row']
    assert len(rows)==2
    assert rows[0].table['cells']==['Ratio','0.48%','Percentage']
    assert rows[1].table['cells'][1]=='460 MPa'
    sections=detect_sections(paper)
    assert [s.title for s in sections]==['Methods','Results']
    chunks=chunk_sections(paper,sections,WideTokenizer(),256,ChunkConfig(180,0,False))
    table_chunks=[c for c in chunks if c.content_kind=='table_row']
    assert len(table_chunks)==2
    assert '0.48%' in table_chunks[0].text and 'Table 1.' in table_chunks[0].repeated_context
    assert 'Meaning (MPa)' in table_chunks[0].repeated_context
    assert table_chunks[0].embedding_input==table_chunks[0].repeated_context+table_chunks[0].text
    from app.pipeline.chunk import section_text
    for section in sections:
        text,pages=section_text(section)
        selected=[c for c in chunks if c.section_path==section.path]
        assert ''.join(c.text for c in selected)==text
        for c in selected:
            assert c.text==text[c.section_char_start:c.section_char_end]
            assert c.page_numbers==sorted({p for p in pages[c.section_char_start:c.section_char_end] if p is not None})
            assert c.token_count<=256
            if c.table and c.content_kind=='table_row':
                for value,(a,b) in zip(c.table['cells'],c.table['cell_spans']): assert text[a:b]==value
    assert chunks==chunk_sections(paper,sections,WideTokenizer(),256,ChunkConfig(180,0,False))


def test_table_continuation(numeric_table_pdf):
    paper=ingest_pdf(numeric_table_pdf(continuation=True))
    rows=[b for b in paper.blocks if b.kind=='table_row']
    assert [b.page for b in rows]==[1,1,2,2]
    assert len({b.table['row_id'] for b in rows})==4
    assert 'Measurements and units' in rows[-1].table['caption'] and 'Cont.' in rows[-1].table['caption']


@pytest.mark.parametrize('value',['0.48%','\u221222.7','1.2e\u22123','460 MPa','+9.4/\u22123.5'])
def test_numeric_safe_fallback(value,tmp_path):
    from app.pipeline.ingest import Paper,TextBlock
    from app.pipeline.chunk import Section
    from app.pipeline.numeric import NUMERIC
    text='word '*12+value+' '+'word '*12+'.'
    block=TextBlock(1,(0,100,400,120),text,[],600,800)
    paper=Paper('hash','synthetic.pdf',1,{},[block],[])
    section=Section('Document',['Document'],[block])
    chunks=chunk_sections(paper,[section],Tokenizer(),35,ChunkConfig(25,0,False))
    assert ''.join(c.text for c in chunks)==text
    assert any(value in c.text for c in chunks)
    boundaries={c.section_char_start for c in chunks}|{c.section_char_end for c in chunks}
    assert all(not any(m.start()<b<m.end() for b in boundaries) for m in NUMERIC.finditer(text))


def test_numeric_table_detector_failure(numeric_table_pdf,monkeypatch):
    import app.pipeline.tables as tables
    monkeypatch.setattr(tables,'detect_tables',lambda page: [])
    paper=ingest_pdf(numeric_table_pdf())
    assert not any(b.table for b in paper.blocks)
    chunks=chunk_sections(paper,detect_sections(paper),Tokenizer(),64,ChunkConfig(35,0,False))
    assert any('0.48%' in c.text for c in chunks)
    assert any('460 MPa' in c.text for c in chunks)


def test_oversized_table_row_and_indivisible_expression(numeric_table_pdf):
    paper=ingest_pdf(numeric_table_pdf(long_cell=True))
    chunks=chunk_sections(paper,detect_sections(paper),WideTokenizer(),200,ChunkConfig(180,0,False))
    pieces=[c for c in chunks if c.content_kind=='table_row' and c.table['cells'][0]=='Strength']
    assert len(pieces)>1 and all(c.warnings for c in pieces)
    assert len({c.table['row_id'] for c in pieces})==1
    assert all('Row: Strength' in c.repeated_context for c in pieces)
    assert '460 MPa' in ''.join(c.text for c in pieces)
    from app.pipeline.numeric import split_safe
    with pytest.raises(ValueError,match='Indivisible'):
        list(split_safe('0.48%',lambda t:len(t)<=3))
    with pytest.raises(ValueError,match='context'):
        chunk_sections(paper,detect_sections(paper),WideTokenizer(),20,ChunkConfig(18,0,False))


def test_embedding_cap_and_lower_model_limit():
    class Larger(Backend):
        max_seq_length=512
        tokenizer=WideTokenizer()
    service=EmbeddingService(model=Larger())
    assert service.token_limit==256
    assert service.model.max_seq_length==512
    with pytest.raises(EmbeddingError): service.embed(['x'*255])
    class Smaller(Larger): max_seq_length=30
    service=EmbeddingService(model=Smaller())
    assert service.token_limit==30
    with pytest.raises(EmbeddingError): service.embed(['x'*29])


@pytest.mark.parametrize('headed',[True,False])
def test_biographies_separated_with_evidence(tmp_path,headed):
    path=tmp_path/'biography.pdf'
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=800)
        page.insert_text((50,100),'References',fontsize=14)
        page.insert_text((50,140),'[1] Jane Doe. Biography of a Professor. University Press, 2020.',fontsize=10)
        page=doc.new_page(width=600,height=800)
        if headed: page.insert_text((50,100),'Author Biographies',fontsize=14)
        page.insert_text((50,150),'Jane Doe',fontname='hebo',fontsize=10)
        page.insert_text((50,175),'She received the Ph.D. degree from Example University.',fontsize=10)
        page.insert_text((50,195),'She is a Professor and conducts research.',fontsize=10)
        doc.save(path)
    sections=detect_sections(ingest_pdf(path))
    assert [s.title for s in sections]==['References','Author Biographies']
    assert all(b.page==2 for b in sections[-1].blocks)
    assert 'Jane Doe' in ' '.join(b.text for b in sections[-1].blocks)


def test_reference_biographical_phrases_and_ambiguous_retention(pdf):
    paper=ingest_pdf(pdf([[(50,100,'References',14),
                          (50,140,'[1] Jane Doe received a Ph.D. degree from University Press.',10),
                          (50,160,'She was a Professor. A biography study, 2020.',10),
                          (50,210,'John received his degree from an institute.',10)]]))
    sections=detect_sections(paper)
    assert [s.title for s in sections]==['References']
    assert any('possible unheaded biography' in w for w in paper.processing_warnings)


def test_structured_table_all_numeric_forms(tmp_path):
    values=['0.48%','\u221222.7','1.2e\u22123','460 MPa','+9.4/\u22123.5']
    path=tmp_path/'numeric-grid.pdf'
    with pymupdf.open() as doc:
        page=doc.new_page(width=600,height=800)
        page.insert_text((50,90),'Methods',fontsize=14)
        page.insert_text((50,130),'Table 1. Numeric measurements.',fontsize=10)
        for y in [150,180,210,240,270,300,330]:page.draw_line((40,y),(560,y))
        for x,t in [(70,'Parameter'),(260,'Value'),(420,'Units')]:page.insert_text((x,170),t,fontsize=10)
        for i,value in enumerate(values):
            y=200+i*30
            for x,t in [(70,'Item '+str(i)),(260,value),(420,'Original')]:
                page.insert_htmlbox((x,y-12,x+130,y+9),t,css='* {font-size:10px;}')
        doc.save(path)
    paper=ingest_pdf(path)
    rows=[b for b in paper.blocks if b.kind=='table_row']
    assert [b.table['cells'][1] for b in rows]==values
    chunks=chunk_sections(paper,detect_sections(paper),WideTokenizer(),256,ChunkConfig(180,0,False))
    for value in values:assert any(value in c.text for c in chunks if c.content_kind=='table_row')


def test_numeric_table_public_serialization_handoff(numeric_table_pdf):
    class WideBackend(Backend):
        tokenizer=WideTokenizer()
        max_seq_length=256
    result=process_pdf(numeric_table_pdf(),service=EmbeddingService(model=WideBackend()),config=ChunkConfig(180,0,False))
    assert result['schema_version']=='1.2'
    payload=vector_store_payload(result)
    assert len({len(v) for v in payload.values()})==1
    for c,metadata in zip(result['chunks'],payload['metadatas']):
        assert metadata['chunk_id']==c['chunk_id']
        if c['table']: assert json.loads(metadata['table'])==c['table']
        assert all(isinstance(v,(str,int,float,bool)) for v in metadata.values())
    json.dumps(result,allow_nan=False)


def test_numeric_expression_across_normalized_whitespace():
    from app.pipeline.numeric import split_safe
    value='460\n\nMPa'
    text='word '*10+value+' '+'word '*10
    pieces=[text[a:b] for a,b in split_safe(text,lambda part:len(part)<=25)]
    assert ''.join(pieces)==text and any(value in piece for piece in pieces)


@pytest.mark.parametrize('value',['100\u00b5g','22.7MPa','9.8m/s\u00b2'])
def test_attached_unit_expressions(value):
    from app.pipeline.numeric import split_safe
    text='words '*6+value+' '+'words '*6
    pieces=[text[a:b] for a,b in split_safe(text,lambda part:len(part)<=20)]
    assert ''.join(pieces)==text and any(value in piece for piece in pieces)


def test_biography_degrees_not_sentence_breaks():
    from app.pipeline.chunk import sentence_spans
    text='Jane received the B.S. and M.S. degrees and the Ph.D. degree. She is a Professor.'
    spans=sentence_spans(text)
    assert [text[a:b].strip() for a,b in spans]==['Jane received the B.S. and M.S. degrees and the Ph.D. degree.','She is a Professor.']


def test_roman_table_caption_not_a_heading(tmp_path):
    path=tmp_path/'caption.pdf'
    with pymupdf.open() as doc:
        page=doc.new_page()
        page.insert_text((50,100),'Methods',fontsize=14)
        page.insert_text((50,150),'Table IV. Measurements',fontname='hebo',fontsize=18)
        page.insert_text((50,190),'A source value is 0.48%.',fontsize=10)
        doc.save(path)
    assert [s.title for s in detect_sections(ingest_pdf(path))]==['Methods']
