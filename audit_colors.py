from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
import re,sys,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'field_extracted/field'))
import field_config as FC
keys=['ground_red','ground_blue','public','start_red','start_blue','ground_red','ground_blue','base_pvc','l1_red','l1_blue','ramp_red','ramp_blue','l1_red','l1_blue','tower_spot','public','transfer_red','transfer_blue','grey','start_red','start_blue','grey','l1_red','l1_blue','grey','tower_spot','base_pvc','grey','base_pvc','base_pvc','block_red','block_blue','block_red','block_blue','stone']
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(next(ROOT.glob('*.docx'))) as z:r=E.fromstring(z.read('word/document.xml'))
rows=[]
for row in r.findall('.//w:tr',ns):
 cells=[''.join(x.text or '' for x in c.findall('.//w:t',ns)) for c in row.findall('w:tc',ns)]
 if len(cells)<3:continue
 m=re.fullmatch(r'(\d+)\s*[-－–]\s*(\d+)\s*[-－–]\s*(\d+)',cells[2])
 if m:rows.append((cells[0],tuple(map(int,m.groups()))))
assert len(rows)==len(keys)
report=[]
for (name,rgb),key in zip(rows,keys):
 assert rgb==FC.RGB[key],(name,rgb,key)
 report.append({'name':name,'rgb':rgb,'config_key':key,'matches':True})
(ROOT/'output/color_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('Rule table: 35/35 RGB entries match shared field_config; rendered shaded pixels are not flat RGB swatches.')
