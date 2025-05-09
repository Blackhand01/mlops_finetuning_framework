"""
Build (or update) an Assistant able to refine training / validation datasets
for the next fine-tuning iteration.
"""

from __future__ import annotations
import logging, textwrap
from typing import Any, Dict, List, Optional
from openai import OpenAI

_LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# internals                                                                   #
# --------------------------------------------------------------------------- #
def _build_system_prompt() -> str:
    return textwrap.dedent(
        """
        You are a senior data-curator specialised in iterative fine-tuning.
        Given:
        • current train / validation JSONL files
        • last fine-tuning metrics (CSV) & loss plot (PNG)
        • list of failed / passed evaluation tests
        analyse the weaknesses of the model and propose JSONL edits
        following **OpenAI chat-format**.

        Your JSON response schema **must** be:
        {
          "action": "patch",          # always
          "add":   [ {messages…}, …], # new examples (optional)
          "edit":  [                 # optional list of patches
              {"index": <int>, "messages": [...]},
              …
          ],
          "remove": [<indices>],      # examples to drop (optional)
          "rationale": "why"
        }
        """
    ).strip()


def _assistant_exists(client: OpenAI, name: str) -> Optional[Dict[str, Any]]:
    for a in client.beta.assistants.list(limit=100, order="asc").data:
        if a.name == name:
            return a
    return None


def _update_assistant(
    client: OpenAI,
    asst_id: str,
    system_prompt: str,
    tool_resources: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    _LOG.info("Updating assistant %s", asst_id)
    return client.beta.assistants.update(
        asst_id,
        instructions=system_prompt,
        tools=[{"type": "code_interpreter"}, {"type": "file_search"}],
        tool_resources=tool_resources,
        model=model,
    ).model_dump()


def _create_assistant(
    client: OpenAI,
    name: str,
    system_prompt: str,
    tool_resources: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    _LOG.info("Creating assistant %s", name)
    return client.beta.assistants.create(
        name=name,
        instructions=system_prompt,
        tools=[{"type": "code_interpreter"}, {"type": "file_search"}],
        tool_resources=tool_resources,
        model=model,
    ).model_dump()


# --------------------------------------------------------------------------- #
# public helper                                                               #
# --------------------------------------------------------------------------- #
def ensure_single_assistant(
    client: OpenAI,
    *,
    lang_pair: str,
    version: str,
    method: str,
    train_file_id: str,
    valid_file_id: str,
    extra_file_ids: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Return an Assistant ready to refine the dataset.

    If an Assistant with the same *name* already exists it is updated, else
    a new one is created.

    Parameters
    ----------
    train_file_id / valid_file_id
        File-IDs just uploaded via Files API.
    extra_file_ids
        IDs of metrics CSV / PNG or evaluation results (optional).
    """
    name = f"{lang_pair}-translator-{version}-{method}-refiner"
    system_prompt = _build_system_prompt()

    file_ids = [train_file_id, valid_file_id] + (extra_file_ids or [])
    tool_resources = {"file_search": {"file_ids": file_ids}}

    existing = _assistant_exists(client, name)
    if existing:
        return _update_assistant(
            client, existing.id, system_prompt, tool_resources, model
        )
    return _create_assistant(client, name, system_prompt, tool_resources, model)
