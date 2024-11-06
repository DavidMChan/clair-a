# CLAIR-A: Leveraging Large Language Models to Judge Audio Captions

Official implementation of the paper: "CLAIR-A: Leveraging Large Language Models to Judge Audio Captions"

## Installation / Getting Started

### Setup

```bash
# (Install the full library)
pip install git+https://github.com/DavidMChan/clair-a.git

# (Temporarily, for OpenAI Structured Generation, PR pending)
pip install outlines@git+https://github.com/lapp0/outlines.git@openai-structured-generation
```

Next, you need to patch the outlines library to use greedy/deterministic sampling with OpenAI. This is temporary until the PR is merged.
Alter the file `outlines/generate/json.py` and replace the `json_openai` function with the following:

```python
# Add a new import
from outlines.samplers import greedy

# Replace the function
@json.register(OpenAI)
def json_openai(
    model, schema_object: Union[str, object, Callable], sampler: Sampler = multinomial()
):
    if not isinstance(sampler, (multinomial, greedy)):
        raise NotImplementedError(
            r"The OpenAI API does not support any other sampling algorithm "
            + "than the multinomial and greedy samplers."
        )

    if isinstance(schema_object, type(BaseModel)):
        schema = pyjson.dumps(schema_object.model_json_schema())
    elif callable(schema_object):
        schema = pyjson.dumps(get_schema_from_signature(schema_object))
    elif isinstance(schema_object, str):
        schema = schema_object
    else:
        raise ValueError(
            f"Cannot parse schema {schema_object}. The schema must be either "
            + "a Pydantic object, a function or a string that contains the JSON "
            + "Schema specification"
        )

    # create copied, patched model with normalized json schema set
    generator = model.new_with_replacements(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "default",
                "strict": True,
                "schema": pyjson.loads(schema),
            },
        },
        # Get the information from the sampler
        temperature=sampler.temperature if isinstance(sampler, multinomial) else 0.0,
        top_p=sampler.top_p if isinstance(sampler, multinomial) else 1.0,
    )

    # set generators sequence post-processor
    if isinstance(schema_object, type(BaseModel)):
        generator.format_sequence = lambda x: schema_object.parse_raw(x)
    elif callable(schema_object):
        generator.format_sequence = lambda x: pyjson.loads(x)
    elif isinstance(schema_object, str) or isinstance(schema_object, dict):
        generator.format_sequence = lambda x: pyjson.loads(x)
    else:
        raise ValueError(
            f"Cannot parse schema {schema_object}. The schema must be either "
            + "a Pydantic object, a function or a string that contains the JSON "
            + "Schema specification"
        )

    return generator
```

Finally, you need to patch the fense library to load checkpoints in a non-strict way:

```python
# In fense/evaluator.py:31
clf.load_state_dict(model_states['state_dict'], strict=False)
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
@article{wu2024clair,
  title={CLAIR-A: Leveraging Large Language Models to Judge Audio Captions},
  author={Wu, Tsung-Han and Gonzalez, Joseph E and Darrell, Trevor and Chan, David M},
  journal={arXiv preprint arXiv:2409.12962},
  year={2024}
}
```
