from __future__ import annotations
from dataclasses import dataclass, asdict, field
import hashlib
import json
import re
from statistics import median
from .ingest import Paper, TextBlock
from .numeric import split_safe, NUMERIC

@dataclass(frozen=True)
class ChunkConfig:
    target_tokens: int = 220
    overlap_tokens: int = 30
    heading_prefix: bool = True
    exclude_references: bool = False
    exclude_appendix: bool = False

@dataclass
class Section:
    title: str
    path: list[str]
    blocks: list[TextBlock]

@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    section_title: str
    section_path: list[str]
    chunk_index: int
    page_start: int
    page_end: int
    page_numbers: list[int]
    text: str
    embedding_input: str
    token_count: int
    section_char_start: int
    section_char_end: int
    warnings: list[str] = field(default_factory=list)
    content_kind: str = 'prose'
    repeated_context: str = ''
    table: dict | None = None

COMMON = re.compile(r'^(Abstract|Introduction|Background|Related Work|Methods?|Methodology|Materials and Methods|Results|Discussion|Conclusions?|References|Bibliography|Appendix|Acknowledg(?:e)?ments?|Keywords|Index Terms)(?:\s+[A-Z])?$', re.I)
BIO_HEADING = re.compile(r'^(?:Author Biograph(?:y|ies)|Biograph(?:y|ies)|About the Authors|Authors? Biography)$', re.I)
NUMBER = re.compile(r'^(\d+(?:\.\d+)*)([.)]?)\s+(.+)$')
ROMAN = re.compile(r'^([IVXLCDM]+)(?:\.(\d+(?:\.\d+)*))?[.)]?\s+(.+)$')
LETTER = re.compile(r'^([A-Z])[.)]\s+(.+)$')
INLINE = re.compile(r'^(Abstract|Keywords|Index Terms)\s*([:—–-])\s*', re.I)
CAPTION = re.compile(r'^(?:Fig(?:ure)?\.?|Table|Algorithm|Equation)\s*(?:\d+|[IVX]+)(?:[.:]|\s|$)', re.I)


def heading_number(text: str, stack: list[str]) -> tuple[str, int, bool]:
    match = NUMBER.fullmatch(text)
    if match:
        level = match[1].count('.') + 1
        if match[2] == ')' and len(stack) >= 2:
            level = 3
        return match[3], level, True
    match = ROMAN.fullmatch(text)
    if match and (len(match[1]) > 1 or match[3].isupper() or match[1] == 'I'):
        # Reject invalid Roman sequences, e.g. an isolated decorative word.
        if re.fullmatch(r'M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})', match[1]):
            return match[3], 1 + (len(match[2].split('.')) if match[2] else 0), True
    match = LETTER.fullmatch(text)
    if match:
        return match[2], 2, True
    return text, 1, False


def section_text(section: Section) -> tuple[str, list[int | None]]:
    parts, pages = [], []
    for i, block in enumerate(section.blocks):
        previous = section.blocks[i-1] if i else None
        separator = '' if not previous else ('\n\n' if block.page != previous.page or block.source_block_index != previous.source_block_index else ' ')
        parts.append(separator + block.text)
        pages.extend([None] * len(separator) + [block.page] * len(block.text))
    return ''.join(parts), pages


def biography_start(rows, index):
    block,line = rows[index]
    if line is None:
        return False
    text = ''.join(s['text'] for s in line['spans']).strip()
    if re.match(r'^\[\d+\]|^\d+[.)]\s', text):
        return False
    previous = rows[index-1] if index else None
    if previous and previous[0].page == block.page and previous[0].source_block_index == block.source_block_index:
        return False
    window = ' '.join(''.join(s['text'] for s in l['spans']) for b,l in rows[index:index+16] if l and b.page==block.page)
    degree = bool(re.search(r'\b(?:received|earned|completed|studied)\b.{0,180}\b(?:degree|Ph\.?D|B\.S|M\.S|physics|engineering)\b', window, re.I))
    role = bool(re.search(r'\b(?:professor|postdoc|researcher|investigator|fellowship|internship)\b', window,re.I))
    institution = bool(re.search(r'\b(?:University|Institute|College|Hospital)\b',window))
    pronoun = bool(re.search(r'\b(?:He|She|His|Her)\b',window))
    bold_name = any(s['flags']&16 and len(s['text'].strip().split())>=2 for s in line['spans'])
    new_page_wide = (not previous or previous[0].page!=block.page) and line['bbox'][2]-line['bbox'][0]>block.page_width*.55
    return degree and institution and (role or pronoun) and (bold_name or new_page_wide)


def detect_sections(paper: Paper) -> list[Section]:
    """Conservative line classifier: lexical identity plus typography and layout."""
    rows = [item for b in paper.blocks for item in ([(b,None)] if b.kind.startswith('table_') else [(b,l) for l in b.lines if any(s['text'].strip() for s in l['spans'])])]
    sizes = [s['size'] for b, line in rows if line and not COMMON.fullmatch(''.join(s['text'] for s in line['spans']).strip())
             for s in line['spans'] if not s['flags'] & 18 for _ in range(max(1, len(s['text'])))]
    body = median(sizes) if sizes else 10
    sections, stack = [], []
    current = Section('Preamble', ['Preamble'], [])
    found = False
    def warn(message):
        if message not in paper.processing_warnings:
            paper.processing_warnings.append(message)
    for row_index, (block, line) in enumerate(rows):
        if line is None:
            current.blocks.append(block)
            continue
        if heading_number(current.title,[])[0].upper() in ('REFERENCES','BIBLIOGRAPHY') and biography_start(rows,row_index):
            sections.append(current)
            stack = ['Author Biographies']
            current = Section('Author Biographies',list(stack),[])
            found = True
            warn(f'Section detection: page {block.page}: inferred Author Biographies from multiple biographical signals and layout/font evidence.')
        text = re.sub(r'[ \t]+', ' ', ''.join(s['text'] for s in line['spans'])).strip()
        spans = line['spans']
        size = max(s['size'] for s in spans)
        styled = any(s['flags'] & 18 for s in spans)  # bold or italic
        bbox = line['bbox']
        margin = bbox[3] < block.page_height * .07 or bbox[1] > block.page_height * .97
        # Use the local column extent, not the page centre, for centred titles.
        same_column = [l['bbox'] for b, l in rows if l and b.page == block.page and
                       (l['bbox'][0] >= block.page_width/2) == (bbox[0] >= block.page_width/2) and
                       l['bbox'][2]-l['bbox'][0] > block.page_width * .25]
        extent = (min(x[0] for x in same_column), max(x[2] for x in same_column)) if same_column else (block.bbox[0], block.bbox[2])
        centred = abs((bbox[0]+bbox[2])/2 - sum(extent)/2) < 12 and bbox[2]-bbox[0] < (extent[1]-extent[0]) * .92
        prior = rows[row_index-1] if row_index else None
        spacing = not prior or prior[1] is None or prior[0].page != block.page or bbox[1] - prior[1]['bbox'][3] > body * .5
        separate = len(block.lines) == 1 or spacing
        title, level, numbered = heading_number(text, stack)
        inline = INLINE.match(text)
        remainder = None
        if inline:
            title, level, numbered = inline[1], 1, False
            remainder = text[inline.end():]
            original_heading = text[:inline.end()].rstrip()
        else:
            original_heading = text
        # Run-in numbered subsections: accept a colon only with styled title spans.
        if not inline and numbered and ':' in text:
            cut = text.index(':') + 1
            prefix_spans, offset = [], 0
            for span in spans:
                if offset < cut:
                    prefix_spans.append(span)
                offset += len(span['text'])
            if prefix_spans and all(s['flags'] & 18 for s in prefix_spans if s['text'].strip()):
                original_heading, remainder = text[:cut], text[cut:].lstrip()
                title, level, numbered = heading_number(original_heading.rstrip(':'), stack)
        short = len(original_heading) <= 140 and len(original_heading.split()) <= 18
        lexical = bool(COMMON.fullmatch(title.rstrip(':—–- ')) or BIO_HEADING.fullmatch(title))
        valid_title = len(title.rstrip(':')) >= 3 and bool(re.search(r'[A-Za-z]{3}', title))
        sentence = original_heading.endswith(('.', '?', '!')) or bool(re.search(r'\b(?:is|are|was|were|has|have|describes|shows)\b', title, re.I))
        furniture = margin or bool(CAPTION.match(text)) or bool(re.match(r'^(?:IEEE TRANSACTIONS|Digital Object Identifier|Copyright|Page\s+\d)', text, re.I))
        evidence = styled or centred or size > body * 1.08 or (spacing and title.isupper())
        custom_title = found and styled and size > body * 1.08 and separate and len(title.split()) >= 2 and all(w[0].isupper() or w.lower() in {'and', 'of', 'the', 'for', 'in'} for w in title.split())
        heading = not furniture and valid_title and short and not sentence and (
            bool(inline) or (lexical and (separate or evidence)) or (numbered and evidence) or custom_title)
        if heading:
            if current.blocks or found:
                sections.append(current)
            stack = stack[:level-1]
            if level > len(stack)+1:
                warn(f'Section detection: page {block.page}: missing parent for {original_heading!r}; available path retained.')
            stack.append(original_heading)
            current, found = Section(original_heading, list(stack), []), True
            if remainder:
                current.blocks.append(TextBlock(block.page, tuple(bbox), remainder, [line], block.page_width, block.page_height, block.source_block_index))
        else:
            current.blocks.append(TextBlock(block.page, tuple(bbox), text, [line], block.page_width, block.page_height, block.source_block_index))
            if not furniture and ((numbered and short and not sentence and title[0].isupper()) or (size > body * 1.5 and len(text) > 2)):
                warn(f'Section detection: page {block.page}: uncertain heading {text!r}; retained as body text.')
            if current.title.upper() in ('REFERENCES', 'BIBLIOGRAPHY') and re.search(r'\breceived (?:the|his|her)\b|\bstudied physics\b', text, re.I):
                warn(f'Section detection: page {block.page}: possible unheaded biography after References; retained in the current section for review.')
    if current.blocks or found:
        sections.append(current)
    if not found:
        for section in sections:
            section.title, section.path = 'Document', ['Document']
        warn('Section detection: no reliable headings; using Document fallback.')
    return sections


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Small deterministic sentence scanner; avoids common abbreviations/initials."""
    ends = []
    for match in re.finditer(r'[.!?][\]"\u201d\')]*\s+', text):
        punct = match.start()
        word = re.search(r'([\w.]+)$', text[:punct])
        previous = word[1].lower() if word else ''
        if text[punct] == '.' and (previous in {'e.g', 'i.e', 'et', 'al', 'fig', 'eq', 'dr', 'mr', 'mrs', 'prof', 'vs', 'approx', 'vol', 'pp', 'no', 'b.s', 'm.s', 'b.a', 'm.d', 'ph.d', 'd.sc', 'm.sc', 'b.sc'} or (len(previous) == 1 and previous.isalpha())):
            continue
        ends.append(match.end())
    ends.append(len(text))
    starts, result = 0, []
    for end in sorted(set(ends)):
        if end > starts:
            result.append((starts, end))
            starts = end
    return result


def chunk_sections(paper: Paper, sections: list[Section], tokenizer, limit: int, config: ChunkConfig = ChunkConfig()) -> list[Chunk]:
    limit = min(limit, int(tokenizer.model_max_length), 256)
    if config.target_tokens <= 0 or config.overlap_tokens < 0 or config.overlap_tokens >= config.target_tokens:
        raise ValueError('Require target_tokens > overlap_tokens >= 0.')
    chunks = []
    for section_index, section in enumerate(sections):
        titles = [heading_number(title.rstrip(':—–- '), [])[0].lower() for title in section.path]
        if (config.exclude_references and any(t in ('references', 'bibliography') for t in titles)) or (config.exclude_appendix and any(t.startswith('appendix') for t in titles)):
            continue
        text, page_map = section_text(section)
        if not text:
            continue
        prefix = ' / '.join(section.path) + '\n\n' if config.heading_prefix else ''
        soft_budget = min(limit, token_count(tokenizer, prefix) + config.target_tokens)
        if token_count(tokenizer, prefix) >= limit:
            raise ValueError('Heading prefix exhausts token budget; disable prefix or increase model limit.')
        units = []
        # Canonical section offsets include paragraph separators and table cell delimiters.
        runs = []
        table_source_offsets = {}
        offset, prose_start = 0, 0
        for i, block in enumerate(section.blocks):
            previous = section.blocks[i-1] if i else None
            separator = '' if not previous else ('\n\n' if block.page != previous.page or block.source_block_index != previous.source_block_index else ' ')
            end = offset + len(separator) + len(block.text)
            if block.kind.startswith('table_'):
                table_source_offsets[id(block)] = offset + len(separator)
                if prose_start < offset:
                    runs.append((prose_start,offset,None))
                runs.append((offset,end,block))
                prose_start = end
            offset = end
        if prose_start < len(text):
            runs.append((prose_start,len(text),None))

        def context(block):
            if block is None:
                return ''
            info = block.table
            result = info['caption'] + '\nHeaders: ' + ' | '.join(info['headers']) + '\n'
            if block.kind == 'table_row':
                result += 'Row: ' + info['cells'][0] + '\n'
            return result

        for run_start,run_end,block in runs:
            spans = [(run_start,run_end)] if block else [(run_start+a,run_start+b) for a,b in sentence_spans(text[run_start:run_end])]
            repeated = context(block)
            full_prefix = prefix + repeated
            if token_count(tokenizer,full_prefix) >= limit:
                raise ValueError('Required table/header/row context or heading prefix cannot fit the effective token limit.')
            for start,end in spans:
                if token_count(tokenizer,full_prefix + text[start:end]) <= limit:
                    units.append((start,end,None,block))
                    continue
                label = 'Oversized table row/header' if block else 'Oversized sentence'
                warning = f'{label} split in {section.title!r}, section characters [{start}, {end}); exact standalone input exceeds {limit} tokens; numeric-safe boundaries used.'
                if warning not in paper.processing_warnings:
                    paper.processing_warnings.append(warning)
                for lo,hi in split_safe(text[start:end],lambda part:token_count(tokenizer,full_prefix+part)<=limit):
                    units.append((start+lo,start+hi,warning,block))
        cursor, overlap_start, index = 0, 0, 0
        while cursor < len(units):
            block = units[cursor][3]
            repeated = context(block)
            full_prefix = prefix + repeated
            first = cursor if block else overlap_start
            while first < cursor and (units[first][3] is not None or token_count(tokenizer,full_prefix + text[units[first][0]:units[cursor][1]]) > soft_budget):
                first += 1
            last = cursor + 1
            # Tables are row units. Prose packs complete sentences, within its own run.
            if block is None:
                while last < len(units) and units[last][3] is None and token_count(tokenizer,full_prefix + text[units[first][0]:units[last][1]]) <= soft_budget:
                    last += 1
                if last < len(units):
                    paragraph_ends = [i for i in range(cursor+1,last+1) if text[:units[i-1][1]].endswith('\n\n')]
                    if paragraph_ends:
                        last = paragraph_ends[-1]
            start,end = units[first][0],units[last-1][1]
            source = text[start:end]
            count = token_count(tokenizer,full_prefix+source)
            if count > limit:
                raise ValueError('Internal chunk token limit violation; truncation is forbidden.')
            pages = sorted({p for p in page_map[start:end] if p is not None})
            if not pages:
                raise ValueError('Token budget produced a separator-only chunk; increase target_tokens.')
            identity = ['numeric-table-v3',paper.paper_id,asdict(config),limit,section_index,section.path,index,start,end]
            cid = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            warnings = list(dict.fromkeys(u[2] for u in units[first:last] if u[2]))
            table = dict(block.table) if block else None
            if table and block.kind == 'table_row':
                cell_start = table_source_offsets[id(block)]
                # Per-cell offsets in canonical section text, never PDF bytes.
                table['cell_spans'] = []
                for cell in table['cells']:
                    table['cell_spans'].append([cell_start,cell_start+len(cell)])
                    cell_start += len(cell)+3
                table['fragment_columns'] = [header for header,(a,b) in zip(table['headers'],table['cell_spans']) if a < end and b > start]
            chunks.append(Chunk(cid,paper.paper_id,section.title,section.path,index,pages[0],pages[-1],pages,source,full_prefix+source,count,start,end,warnings,block.kind if block else 'prose',repeated,table))
            index += 1
            cursor = last
            overlap_start = cursor
            if block is None:
                for candidate in range(cursor-1,first,-1):
                    if units[candidate][3] is not None or units[candidate][2] or token_count(tokenizer,text[units[candidate][0]:end])-token_count(tokenizer,'') > config.overlap_tokens:
                        break
                    overlap_start = candidate
    return chunks
