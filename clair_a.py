import random
from typing import Annotated, List, Literal, Optional, Tuple, Union

from openai import OpenAI
from outlines.models.openai import from_openai as outlines_model_from_openai
from outlines.models.transformers import from_transformers as outlines_model_from_transformers
from pydantic import BaseModel, ConfigDict, conint, constr
from transformers import AutoModelForCausalLM, AutoTokenizer

from fense.evaluator import Evaluator

# Lazy initialize the FENSE evaluator and metrics
_engine_cache = {}
_fense_evaluator = None


_CLAIRA_PROMPT = """\
You are tasked with evaluating if a set of candidate captions accurately describes the same sound in a video clip as a reference set of captions. Start by assessing the accuracy and precision of how the audio characteristics are captured in the captions, scoring from 0 to 90 based on this aspect alone. After this initial assessment, you may add additional points (from 0 to 10) based on the quality of grammar and the detailed, reasonable descriptions present in the captions.

Candidate set:
{candidate_statements}

Reference set:
{target_statements}

Combine these two aspects for a final evaluation score on a scale from 0 to 100, reflecting the likelihood that the candidate set is describing the same sound as the reference set. Format your response in JSON with a key "score", value between 0 and 100, and a key "reason" with a string value explaining your assessment.
"""


class CLAIRAResponse(BaseModel):
    score: Annotated[int, conint(ge=0, le=100)]
    reason: Annotated[str, constr(max_length=1024)]


class CLAIRAReponseOpenAI(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: int
    reason: str


def _engine_from_cache(model: str) -> Tuple[callable, Union[type[CLAIRAResponse], type[CLAIRAReponseOpenAI]]]:
    # Initialize the generator using outlines
    if model not in _engine_cache:
        if model.startswith("openai/"):
            # Use new outlines API for OpenAI models
            model_name = model[len("openai/") :]
            client = OpenAI()  # Uses OPENAI_API_KEY environment variable
            outlines_model = outlines_model_from_openai(client, model_name)
            response_type = CLAIRAReponseOpenAI
        elif model.startswith("transformers/"):
            # Use new outlines API for transformers models
            model_name = model[len("transformers/") :]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            outlines_model = outlines_model_from_transformers(hf_model, tokenizer)
            response_type = CLAIRAResponse
        else:
            raise ValueError(
                f"Unknown model: {model} (Prefix openai models with 'openai/', transformers models with 'transformers/')"
            )

        _engine_cache[model] = (outlines_model, response_type)
    else:
        outlines_model, response_type = _engine_cache[model]

    return outlines_model, response_type


def clair_a(
    candidate: str,
    targets: List[str],
    model: str = "openai/gpt-4o-2024-08-06",
    tiebreaking_epsilon: float = 0.0001,
    tiebreaking_method: Union[Literal["fense"], Literal["random"]] = "fense",
) -> Tuple[float, Optional[str]]:

    # Get the outlines model
    outlines_model, response_type = _engine_from_cache(model)

    # Format the candidates and targets
    candidate_statements = [f"- {c}\n" for c in [candidate]]
    target_statements = [f"- {t}\n" for t in targets]
    formatted_prompt = _CLAIRA_PROMPT.format(
        candidate_statements="".join(candidate_statements),
        target_statements="".join(target_statements),
    )

    # Use new outlines API - call model directly with output_type
    if model.startswith("openai/"):
        response_json = outlines_model(formatted_prompt, response_type)
    elif model.startswith("transformers/"):
        response_json = outlines_model(formatted_prompt, response_type, max_new_tokens=1024)
    else:
        raise ValueError(
            f"Incompatible model: {model} (Prefix openai models with 'openai/', transformers models with 'transformers/')"
        )

    # Parse the response using Pydantic if it's a JSON string
    if isinstance(response_json, str):
        response = response_type.model_validate_json(response_json)
    elif isinstance(response_json, response_type):
        # If it's already a Pydantic model instance
        response = response_json
    else:
        raise ValueError(f"Unexpected response format: {response_json}")

    # Add the tiebreaking score
    if tiebreaking_method == "fense":
        if _engine_cache.get("_fense_evaluator") is None:
            _engine_cache["_fense_evaluator"] = Evaluator(
                device="cpu", sbert_model="paraphrase-mpnet-base-v2", echecker_model="echecker_clotho_audiocaps_tiny"
            )
        tiebreaking_score, _, _ = _engine_cache["_fense_evaluator"].sentence_score(
            candidate, targets, return_error_prob=True
        )
    elif tiebreaking_method == "random":
        tiebreaking_score = random.uniform(0, 1)

    overall_score = (response.score / 100) + tiebreaking_epsilon * tiebreaking_score

    return overall_score, response.reason


if __name__ == "__main__":
    candidate = "Rain is splashing on a surface while rustling occurs and a car door shuts, and traffic is discernible in the distance"
    references = [
        "Rain falls soft and steadily and a person closes a car door and walks away through leaves",
        "Rain falling followed by fabric rustling and footsteps shuffling then a vehicle door opening and closing as plastic crinkles",
        "Rain falling followed by footsteps walking on grass then a vehicle door opening then closing",
        "Light rainfall together with rustling",
    ]

    # score = clair_a(candidate, references, model="openai/gpt-4o-2024-08-06")
    score = clair_a(candidate, references, model="transformers/microsoft/Phi-4-mini-instruct")
    print(score)
