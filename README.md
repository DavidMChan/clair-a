# CLAIR-A: Leveraging Large Language Models to Judge Audio Captions

Official implementation of the paper: "CLAIR-A: Leveraging Large Language Models to Judge Audio Captions"

## Installation / Getting Started

### Setup

```bash
# (Install the full library)
pip install git+https://github.com/DavidMChan/clair-a.git
```

### Usage

```python
import os
from clair_a import clair_a

# If using OpenAI
os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

candidates = ['Rain is splashing on a surface while rustling occurs and a car door shuts, and traffic is discernible in the distance']
references = ['Rain falls soft and steadily and a person closes a car door and walks away through leaves',
              'Rain falling followed by fabric rustling and footsteps shuffling then a vehicle door opening and closing as plastic crinkles',
              'Rain falling followed by footsteps walking on grass then a vehicle door opening then closing',
              'Light rainfall together with rustling']

score = clair_a(candidates, references, model='openai/gpt-4o-2024-08-06')
print(score)
# (0.78, 'The candidate caption captures the main elements of the soundscape described in the reference set, such as rain, rustling, and a car door shutting. However, it lacks some of the nuanced details present in the reference captions, such as footsteps and the specific sequence of sounds. The candidate caption is concise and grammatically correct, but it could benefit from more detailed descriptions to match the precision of the reference set. Overall, the candidate caption is a good match but could be improved with additional detail and specificity.')

score = clair_a(candidates, references, model='transformers/microsoft/Phi-4-mini-instruct')
print(score)
# (0.6500784354805946, "The candidate set captures the essence of rain and the sound of a car door shutting, which are present in the reference set. However, it lacks the detailed descriptions of the rain's softness, the rustling of fabric, and the specific sounds of footsteps and plastic crinkles. The grammar is generally good, but the descriptions could be more detailed and precise.")
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{wu2024clair,
  title={CLAIR-A: Leveraging Large Language Models to Judge Audio Captions},
  author={Wu, Tsung-Han and Gonzalez, Joseph E and Darrell, Trevor and Chan, David M},
  journal={arXiv preprint arXiv:2409.12962},
  year={2024}
}
```
