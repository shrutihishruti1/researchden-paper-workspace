from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
import hashlib
import re
import pymupdf

class IngestionError(ValueError):
    """PDF cannot be processed without intervention."""

@dataclass(frozen=True)
class TextBlock:
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    lines: list[dict]
    page_width: float
    page_height: float
    source_block_index: int = 0
    kind: str = 'prose'
    table: dict | None = None

@dataclass
class Paper:
    paper_id: str
    source_filename: str
    page_count: int
    metadata: dict
    blocks: list[TextBlock]
    warnings: list[str]
    processing_warnings: list[str] = field(default_factory=list)

def reading_order(blocks: list[TextBlock]) -> list[TextBlock]:
    """Column bands separated by full-width blocks; each band reads left then right."""
    if not blocks:
        return []
    middle = blocks[0].page_width / 2
    left = [b for b in blocks if b.bbox[2] <= middle + 8]
    right = [b for b in blocks if b.bbox[0] >= middle - 8]
    if not left or not right or set(map(id, left)) & set(map(id, right)):
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    spanning = sorted([b for b in blocks if b not in left and b not in right], key=lambda b: b.bbox[1])
    result, remaining = [], left + right
    for separator in spanning:
        band = [b for b in remaining if b.bbox[1] < separator.bbox[1]]
        result.extend(sorted(band, key=lambda b: (b.bbox[0] >= middle - 8, b.bbox[1], b.bbox[0])))
        remaining = [b for b in remaining if b not in band]
        result.append(separator)
    return result + sorted(remaining, key=lambda b: (b.bbox[0] >= middle - 8, b.bbox[1], b.bbox[0]))

def ingest_pdf(source: str | Path | bytes, source_filename: str | None = None) -> Paper:
    if isinstance(source, bytes):
        data, filename = source, source_filename or 'document.pdf'
    else:
        path = Path(source)
        data, filename = path.read_bytes(), source_filename or path.name
    blocks, warnings = [], []
    try:
        with pymupdf.open(stream=data, filetype='pdf') as doc:
            if doc.needs_pass:
                raise IngestionError('Password-protected PDF: supply an unencrypted PDF.')
            count, metadata = len(doc), {k: v for k, v in doc.metadata.items() if v}
            table_captions = {}
            for number, page in enumerate(doc, 1):
                page_blocks = []
                try:
                    from .tables import detect_tables
                    tables = detect_tables(page)
                except Exception as exc:
                    tables = []
                    warnings.append(f'Page {number}: table detection unavailable; text retained as numeric-safe fallback ({type(exc).__name__}).')
                from .tables import CAPTION
                caption_ids = {m[1] for b in page.get_text('dict', flags=pymupdf.TEXTFLAGS_TEXT)['blocks'] if b['type']==0 for l in b['lines'] if (m := CAPTION.match(''.join(s['text'] for s in l['spans']).strip()))}
                uncertain = caption_ids - {t['id'] for t in tables}
                if uncertain:
                    warnings.append(f'Page {number}: table(s) {", ".join(sorted(uncertain))} lack reliable row/cell geometry; source retained as numeric-safe fallback.')
                for block_index, raw in enumerate(page.get_text('dict', flags=pymupdf.TEXTFLAGS_TEXT)['blocks']):
                    if raw['type'] != 0:
                        continue
                    lines = raw['lines']
                    lines = [l for l in lines if not any(t['bbox'][1] <= (l['bbox'][1]+l['bbox'][3])/2 < t['bbox'][3] and t['bbox'][0] <= (l['bbox'][0]+l['bbox'][2])/2 <= t['bbox'][2] for t in tables)]
                    if not lines:
                        continue
                    middle = page.rect.width / 2
                    groups = [[l for l in lines if l['bbox'][2] <= middle + 8],
                              [l for l in lines if l['bbox'][0] >= middle - 8],
                              [l for l in lines if l['bbox'][2] > middle + 8 and l['bbox'][0] < middle - 8]]
                    # MuPDF can merge opposite-column lines into one text block.
                    # Split that block by line geometry before ordering the page.
                    groups = groups if groups[0] and groups[1] else [lines]
                    for group in groups:
                        if not group:
                            continue
                        text = '\n'.join(re.sub(r'[ \t]+', ' ', ''.join(s['text'] for s in line['spans'])).strip() for line in group).strip()
                        bbox = (min(l['bbox'][0] for l in group), min(l['bbox'][1] for l in group),
                                max(l['bbox'][2] for l in group), max(l['bbox'][3] for l in group))
                        if text:
                            page_blocks.append(TextBlock(number, bbox, text, group, page.rect.width, page.rect.height, block_index))
                for table in tables:
                    tid = table['id']
                    caption = table['caption']
                    if 'cont' in caption.lower() and tid in table_captions:
                        caption = table_captions[tid] + ' / ' + caption
                    else:
                        table_captions[tid] = caption
                    common = {'table_id': tid, 'caption': caption, 'headers': table['headers'], 'method': table['method']}
                    header = ' | '.join(table['headers'])
                    page_blocks.append(TextBlock(number, (table['bbox'][0],table['bbox'][1],table['bbox'][2],table['rows'][0]['bbox'][1]), header, table['header_lines'], page.rect.width,page.rect.height,-1,'table_header',common))
                    for index,row in enumerate(table['rows']):
                        info = dict(common, row_id=f'{tid}:p{number}:r{index}', cells=row['cells'])
                        page_blocks.append(TextBlock(number,row['bbox'],' | '.join(row['cells']),row['lines'],page.rect.width,page.rect.height,-2-index,'table_row',info))
                if not page_blocks:
                    warnings.append(f'Page {number} has no extractable text; OCR may be required.')
                blocks.extend(reading_order(page_blocks))
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f'Malformed or unreadable PDF: {exc}') from exc
    if not blocks:
        raise IngestionError('No extractable text; OCR is required for image-only PDFs.')
    def line_text(line):
        return re.sub(r'[ \t]+', ' ', ''.join(s['text'] for s in line['spans'])).strip()
    def margin_line(b, line):
        return line['bbox'][3] < b.page_height * .07 or line['bbox'][1] > b.page_height * .93
    pages = {}
    sizes = []
    for b in blocks:
        if b.kind.startswith('table_'):
            continue
        for line in b.lines:
            if margin_line(b, line):
                pages.setdefault(line_text(line), set()).add(b.page)
            else:
                sizes.extend(s['size'] for s in line['spans'] for _ in range(max(1, len(s['text']))))
    body_size = median(sizes) if sizes else 10
    retained = []
    for b in blocks:
        if b.kind.startswith('table_'):
            retained.append(b)
            continue
        kept = []
        for line in b.lines:
            text = line_text(line)
            repeats = len(pages.get(text, ()))
            small_alternating = max(s['size'] for s in line['spans']) <= body_size * .85 and repeats >= max(3, count // 2)
            repeated = repeats >= max(3, count * .6) or small_alternating
            if margin_line(b, line) and (re.fullmatch(r'(?:Page\s+)?\d+', text, re.I) or repeated):
                warnings.append(f'Removed margin text on page {b.page}: {text}')
            else:
                kept.append(line)
        if kept:
            bbox = (min(l['bbox'][0] for l in kept), min(l['bbox'][1] for l in kept), max(l['bbox'][2] for l in kept), max(l['bbox'][3] for l in kept))
            retained.append(TextBlock(b.page, bbox, '\n'.join(line_text(l) for l in kept), kept, b.page_width, b.page_height, b.source_block_index))
    if not retained:
        raise IngestionError('No body text remains after margin filtering; OCR may be required.')
    return Paper(hashlib.sha256(data).hexdigest(), filename, count, metadata, retained, warnings)
