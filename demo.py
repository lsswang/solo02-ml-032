import json
from pathlib import Path
from loss import loss
from visible import decode
if __name__=='__main__':
 d=json.loads((Path(__file__).parent/'fixtures/overlap.json').read_text());print(json.dumps({'loss':loss(d['class_logits'],d['mask_logits'],d['depth'],d['targets'],d['beta'],d['gamma']),'visible':decode(d['class_logits'],d['mask_logits'],d['depth'])},indent=2))
