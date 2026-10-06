"""Conservative caption-anchored horizontal-rule table reconstruction."""
import re
import pymupdf

CAPTION = re.compile(r'^Table\s+(\d+|[IVX]+)[.:]', re.I)

def detect_tables(page):
    raw = page.get_text('dict', flags=pymupdf.TEXTFLAGS_TEXT)
    lines = [l for b in raw['blocks'] if b['type']==0 for l in b['lines']]
    owners = {id(l): b['lines'] for b in raw['blocks'] if b['type']==0 for l in b['lines']}
    text = lambda l: ''.join(s['text'] for s in l['spans']).strip()
    captions = [l for l in lines if CAPTION.match(text(l))]
    # The native detector is used for full grids; caption/rule geometry also
    # supports horizontal-only rules, which the native default misses.
    native = page.find_tables().tables
    rules = []
    rule_bounds = {}
    for drawing in page.get_drawings():
        for item in drawing['items']:
            if item[0]=='l' and abs(item[1].y-item[2].y)<.2 and abs(item[1].x-item[2].x)>page.rect.width*.4:
                y=round(item[1].y,2)
                rules.append(y)
                rule_bounds[y]=(min(item[1].x,item[2].x),max(item[1].x,item[2].x))
    rules=sorted(set(rules))
    result=[]
    for caption in captions:
        label=text(caption); next_caption=min([l['bbox'][1] for l in captions if l['bbox'][1]>caption['bbox'][3]] or [page.rect.height*.94])
        below=[y for y in rules if caption['bbox'][3]<y<next_caption]
        if len(below)<3 or below[0]-caption['bbox'][3]>35:
            continue
        top,header_end=below[:2]
        label = ' '.join(text(l) for l in owners[id(caption)] if caption['bbox'][1] <= l['bbox'][1] < top)
        left,right=rule_bounds[top]
        header_lines=[l for l in lines if top <= (l['bbox'][1]+l['bbox'][3])/2 < header_end and left <= (l['bbox'][0]+l['bbox'][2])/2 <= right]
        clusters=[]
        for l in sorted(header_lines,key=lambda l:(l['bbox'][1],l['bbox'][0])):
            center=(l['bbox'][0]+l['bbox'][2])/2
            match=next((c for c in clusters if abs(c[0]-center)<14),None)
            if match: match[1].append(l)
            else: clusters.append([center,[l]])
        clusters.sort(key=lambda c:c[0])
        if not 2<=len(clusters)<=12:
            continue
        centers=[c[0] for c in clusters]; headers=[' '.join(text(l) for l in c[1]) for c in clusters]
        rows=[]
        for lo,hi in zip(below[1:],below[2:]):
            if hi-lo>100: break  # Do not swallow a following prose region.
            row_lines=[l for l in lines if lo <= (l['bbox'][1]+l['bbox'][3])/2 < hi and left <= (l['bbox'][0]+l['bbox'][2])/2 <= right]
            if not row_lines: break
            cells=[[] for _ in centers]
            for l in sorted(row_lines,key=lambda l:(l['bbox'][1],l['bbox'][0])):
                # Assign intact spans by geometry; never clip text at column edges.
                per_cell={}
                for s in l['spans']:
                    idx=min(range(len(centers)),key=lambda i:abs((s['bbox'][0]+s['bbox'][2])/2-centers[i]))
                    per_cell.setdefault(idx,[]).append(s['text'])
                for idx,values in per_cell.items(): cells[idx].append(''.join(values).strip())
            values=[' '.join(v for v in cell if v) for cell in cells]
            if sum(bool(v) for v in values)<2: break
            rows.append({'cells':values,'bbox':(left,lo,right,hi),'lines':row_lines})
        if not rows: continue
        end=rows[-1]['bbox'][3]
        result.append({'id':CAPTION.match(label)[1], 'caption':label, 'headers':headers,'header_lines':header_lines,
                       'bbox':(left,top,right,end),'rows':rows,'method':'caption-horizontal-rules',
                       'native_candidates':len(native)})
    return result
