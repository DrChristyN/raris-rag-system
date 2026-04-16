import json


with open('data/processed/chunks.json', encoding='utf-8') as f:
   chunks = json.load(f)
bspc = [
   c for c in chunks
   if 'BSPC' in c['source_file']
][:3]


for i, c in enumerate(bspc):
   print(f"--- Chunk {i+1} ---")
   print(f"Page: {c['page_number']}")
   print(f"Size: {c['chunk_size']} chars")
   print(f"Text: {c['text'][:200]}")
   print()