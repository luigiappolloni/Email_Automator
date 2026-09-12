import json
import os
import random
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


class InternalRecord(BaseModel):
    code: str = Field(
        description="Policy or claim identifier received in the input"
    )

    record_type: Literal[
        "policy",
        "claim",
    ] = Field(
        description="Whether the record represents a policy or a claim"
    )

    insurance_type: Literal[
        "auto",
        "home",
        "travel",
        "unknown",
    ] = Field(
        description="Insurance product associated with the record"
    )

    status: Literal[
        "active",
        "expired",
        "cancelled",
        "payment_pending",
        "received",
        "under_review",
        "waiting_for_documents",
        "assessment_in_progress",
        "payment_processing",
        "closed",
    ] = Field(
        description="Current internal status of the policy or claim"
    )

    details: str = Field(
        description=(
            "Short internal description containing information "
            "that may be useful when handling a customer email"
        )
    )


class InternalDatabase(BaseModel):
    records: list[InternalRecord] = Field(
        description="Synthetic internal policy and claim records"
    )


POLICY_STATUSES = {
    "active",
    "expired",
    "cancelled",
    "payment_pending",
}

CLAIM_STATUSES = {
    "received",
    "under_review",
    "waiting_for_documents",
    "assessment_in_progress",
    "payment_processing",
    "closed",
}


SYSTEM_PROMPT = """
You generate a small synthetic internal database for an insurance
email automation project.

You will receive a JSON list containing policy and claim identifiers.

For every input item, generate exactly one internal database record.

Important rules:
- Preserve every code exactly as provided.
- Preserve the provided record_type.
- Do not invent new identifiers.
- Do not omit identifiers.
- Do not create duplicate records.
- All generated information must be fictional.
- Keep the details concise, operational, and realistic.
- Do not include customer names or other personal information.
- Do not promise claim approval, reimbursement, or a specific outcome.

Insurance type rules:
- AUT references will normally be auto insurance.
- HMO references will normally be home insurance.
- TRV references will normally be travel insurance.
- CLM references may require the supplied context or may be unknown.
- HCL references will normally be home insurance claims.
- TRC references will normally be travel insurance claims.
- The record_type supplied in the input is authoritative, regardless
  of the identifier prefix.

Policy records can only use these statuses:
- active
- expired
- cancelled
- payment_pending

Claim records can only use these statuses:
- received
- under_review
- waiting_for_documents
- assessment_in_progress
- payment_processing
- closed

Examples of useful policy details:
- Standard cover is currently active.
- Renewal is due within thirty days.
- Payment verification is pending.
- The policy expired before the latest customer request.

Examples of useful claim details:
- Initial documents have been received.
- A repair estimate is still missing.
- The claim is awaiting assessment.
- Payment processing has started.
- The claim is closed and requires human review for further queries.
"""


def load_classified_emails(
    input_path: Path,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Classified email dataset not found: {input_path}"
        )

    dataframe = pd.read_csv(input_path)

    required_columns = {
        "policy_number",
        "claim_number",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "The classified dataset is missing these columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def clean_code(value: object) -> str | None:
    if pd.isna(value):
        return None

    code = str(value).strip()

    if not code:
        return None

    if code.lower() in {
        "none",
        "null",
        "nan",
    }:
        return None

    return code


def extract_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for value in dataframe["policy_number"]:
        code = clean_code(value)

        if code is None:
            continue

        key = (code, "policy")

        if key not in seen:
            records.append(
                {
                    "code": code,
                    "record_type": "policy",
                }
            )
            seen.add(key)

    for value in dataframe["claim_number"]:
        code = clean_code(value)

        if code is None:
            continue

        key = (code, "claim")

        if key not in seen:
            records.append(
                {
                    "code": code,
                    "record_type": "claim",
                }
            )
            seen.add(key)

    return records


def select_records(
    records: list[dict[str, str]],
    inclusion_rate: float,
    random_seed: int,
) -> list[dict[str, str]]:
    if not 0 < inclusion_rate <= 1:
        raise ValueError(
            "inclusion_rate must be greater than 0 "
            "and less than or equal to 1."
        )

    if not records:
        raise ValueError(
            "No policy or claim codes were found."
        )

    random_generator = random.Random(
        random_seed
    )

    selected_records: list[dict[str, str]] = []

    for record_type in ("policy", "claim"):
        type_records = [
            record
            for record in records
            if record["record_type"] == record_type
        ]

        if not type_records:
            continue

        number_to_select = max(
            1,
            round(len(type_records) * inclusion_rate),
        )

       