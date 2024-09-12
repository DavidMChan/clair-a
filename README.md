# CLAIR-A: Leveraging Large Language Models to Judge Audio Captions

Official implementation of the paper: "CLAIR-A: Leveraging Large Language Models to Judge Audio Captions"

Code for the paper will be released soon.

## Installation / Getting Started

### Setup

```bash
# (Temporarily, for OpenAI Structured Generation, PR pending)
pip install outlines@git+https://github.com/lapp0/outlines.git@openai-structured-generation

# (Install the full library)
pip install git+https://github.com/DavidMChan/clair-a.git
```

### Usage

```python
import os
from clair_a import clair_a

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

candidates = ['Rain is splashing on a surface while rustling occurs and a car door shuts, and traffic is discernible in the distance']
references = ['Rain falls soft and steadily and a person closes a car door and walks away through leaves',
              'Rain falling followed by fabric rustling and footsteps shuffling then a vehicle door opening and closing as plastic crinkles',
              'Rain falling followed by footsteps walking on grass then a vehicle door opening then closing',
              'Light rainfall together with rustling']

score = clair_a(candidates, references, model='openai/gpt-4o-2024-08-06')
print(score)
# (0.78, 'The candidate caption captures the main elements of the soundscape described in the reference set, such as rain, rustling, and a car door shutting. However, it lacks some of the nuanced details present in the reference captions, such as footsteps and the specific sequence of sounds. The candidate caption is concise and grammatically correct, but it could benefit from more detailed descriptions to match the precision of the reference set. Overall, the candidate caption is a good match but could be improved with additional detail and specificity.')
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex
[To be updated]
```
