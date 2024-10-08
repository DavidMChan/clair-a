from typing import List, Optional, Tuple

import outlines
from fense.evaluator import Evaluator
from outlines import models
from outlines.samplers import greedy
from pydantic import BaseModel, ConfigDict, conint, constr

_engine_cache = {}
_evaluator = Evaluator(
    device="cpu", sbert_model="paraphrase-mpnet-base-v2", echecker_model="echecker_clotho_audiocaps_tiny"
)

_CLAIRA_PROMPT = """\
You are tasked with evaluating if a set of candidate captions accurately describes the same sound in a video clip as a reference set of captions. Start by assessing the accuracy and precision of how the audio characteristics are captured in the captions, scoring from 0 to 90 based on this aspect alone. After this initial assessment, you may add additional points (from 0 to 10) based on the quality of grammar and the detailed, reasonable descriptions present in the captions.

Candidate set:
{candidate_statements}

Reference set:
{target_statements}

Combine these two aspects for a final evaluation score on a scale from 0 to 100, reflecting the likelihood that the candidate set is describing the same sound as the reference set. Format your response in JSON with a key "score", value between 0 and 100, and a key "reason" with a string value explaining your assessment.
"""


class CLAIRAResponse(BaseModel):
    score: conint(ge=0, le=100)
    reason: constr(max_length=1024)


class CLAIRAReponseOpenAI(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: int
    reason: str


_RESPONSE_TYPE = CLAIRAResponse


def clair_a(
    candidate: str,
    targets: List[str],
    model: str = "openai/gpt-4o-2024-08-06",
    epsilon: float = 0.0001,
) -> Tuple[float, Optional[str]]:
    # Compute the CLAIR-A score for a list of candidates and targets.
    global _engine_cache, _RESPONSE_TYPE  # noqa

    # Initialize the generator using outlines
    if model not in _engine_cache:
        if model.startswith("openai/"):
            outlines_model = models.openai(model[len("openai/") :])
            _RESPONSE_TYPE = CLAIRAReponseOpenAI
        elif model.startswith("transformers/"):
            outlines_model = models.transformers(model[len("transformers/") :], device="cuda")
        else:
            raise ValueError(
                f"Unknown model: {model} (Prefix openai models with 'openai/', transformers models with 'transformers/')"
            )

        generator = outlines.generate.json(outlines_model, _RESPONSE_TYPE, sampler=greedy())

        _engine_cache[model] = generator
    else:
        generator = _engine_cache[model]

    # Format the canndidates and targets
    candidate_statements = [f"- {c}\n" for c in [candidate]]
    target_statements = [f"- {t}\n" for t in targets]
    formatted_prompt = _CLAIRA_PROMPT.format(
        candidate_statements="".join(candidate_statements),
        target_statements="".join(target_statements),
    )

    response = generator(formatted_prompt)

    # Add the tiebreaking score
    score, _, _ = _evaluator.sentence_score(candidate, targets, return_error_prob=True)
    overall_score = (1 - epsilon) * (response.score / 100) + epsilon * score

    return overall_score, response.reason


if __name__ == "__main__":
    candidate = "Rain is splashing on a surface while rustling occurs and a car door shuts, and traffic is discernible in the distance"
    references = [
        "Rain falls soft and steadily and a person closes a car door and walks away through leaves",
        "Rain falling followed by fabric rustling and footsteps shuffling then a vehicle door opening and closing as plastic crinkles",
        "Rain falling followed by footsteps walking on grass then a vehicle door opening then closing",
        "Light rainfall together with rustling",
    ]

    score = clair_a(candidate, references, model="openai/gpt-4o-2024-08-06")
    print(score)
