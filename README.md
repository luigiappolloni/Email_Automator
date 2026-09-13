# Insurance Email Agent

An end-to-end multi-agent prototype for processing customer emails received by an insurance company.

The project generates a synthetic email dataset, classifies each request, retrieves relevant internal policy or claim information, and either:

- generates a professional response for the customer; or
- escalates the request to a human operator with a short explanation.

A final Evaluation Agent assesses the coherence of the automatically generated replies.

## Project Overview

The pipeline is composed of the following stages:

```text
Synthetic insurance emails
            |
            v
     Classifier Agent
            |
            v
 classified_emails.csv
            |
            v
 Case Management Agent
      /             \
     v               v
Policy/claim      Human escalation
retrieval         or customer reply
      \             /
       v           v
    case_decisions.csv
            |
            v
     Evaluation Agent
            |
            v
 coherent / not_coherent
```

## Main Components

### 1. Data Generation

The project starts from a synthetic dataset of customer emails that an insurance company could receive.

Each email contains information such as:

- email identifier;
- subject;
- body;
- insurance type;
- policy number, when available;
- claim number, when available.

Synthetic internal policy and claim records are also generated and stored locally for use by the Case Management Agent.

### 2. Classifier Agent

The Classifier Agent analyzes each customer email and produces structured information, including:

- insurance type;
- customer intent;
- priority;
- policy number;
- claim number;
- whether human review may be required;
- classification confidence.

The classified requests are saved in:

```text
data/classified_emails.csv
```

### 3. Case Management Agent

The Case Management Agent receives one classified email at a time.

When necessary, it uses function tools to retrieve a policy or claim from the internal records. It never invents internal information and does not claim that a record exists unless it has been found.

For every request, the agent selects one of two actions:

```text
reply_to_client
```

or:

```text
escalate_to_human
```

If sufficient verified information is available, the agent generates a professional customer response. If the request is sensitive, incomplete, disputed, unsupported by the internal records, or otherwise unsuitable for automatic handling, it generates a note for a human operator.

The results are saved in:

```text
data/case_decisions.csv
```

The output includes the original request, its classification, the selected action, the decision reason, the generated response, or the note for the human operator.

### 4. Evaluation Agent

The Evaluation Agent selects only the rows for which the Case Management Agent generated an automatic reply.

It compares:

- the original customer email;
- the automated response.

Each response is classified as:

```text
coherent
```

or:

```text
not_coherent
```

The current evaluation focuses on whether the reply is relevant, clear, professional, sufficiently complete, and consistent with the customer's request.

## Results

The complete pipeline was executed on 150 synthetic insurance emails.

### Case Management Results

- Total emails processed: **150**
- Automatic replies generated: **85**
- Requests escalated to a human: **65**
- Automatic reply rate: **56.7%**
- Human escalation rate: **43.3%**

### Evaluation Results

All 85 automatically generated replies were evaluated successfully.

- Replies evaluated: **85**
- Coherent replies: **60**
- Not coherent replies: **25**
- Evaluation errors: **0**
- Coherence rate: **70.6%**
- Non-coherence rate: **29.4%**

These results provide a baseline for improving the Case Management Agent. In particular, the 25 non-coherent responses can be inspected to identify recurring failure modes and refine the prompts, escalation rules, and evaluation criteria.

## Project Structure

```text
insurance_email_agent/
├── data/
│   ├── classified_emails.csv
│   ├── internal_records.csv
│   └── case_decisions.csv
├── case_management_agent.py
├── evaluation_agent.py
├── pyproject.toml
├── README.md
├── .env
└── .gitignore
```

Additional data-generation and classification scripts may be included depending on the local project organization.

## Technologies

- Python
- pandas
- Pydantic
- OpenAI Responses API
- Structured Outputs
- Function calling
- python-dotenv
- uv

## Installation

Clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd insurance_email_agent
uv sync
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5.4-mini
MAX_EMAILS=5
```

`MAX_EMAILS` can be used during testing to limit the number of processed emails. Set it to the complete dataset size or leave it empty, according to the implementation, to process all available requests.

## Usage

Run the Case Management Agent:

```bash
uv run case_management_agent.py
```

It generates:

```text
data/case_decisions.csv
```

Then run the Evaluation Agent:

```bash
uv run evaluation_agent.py
```

The terminal prints the total number of coherent responses, non-coherent responses, and evaluation errors.

## Safety and Decision Rules

The Case Management Agent follows several constraints:

- it does not invent policy or claim information;
- it retrieves internal information through dedicated tools;
- it does not approve or reject claims;
- it does not promise payments or reimbursements;
- it does not modify policies or claims;
- it escalates requests that cannot be handled safely and completely;
- it provides a useful note to the human operator when escalation is required.

## Limitations

This project is a prototype built with synthetic data.

The current Evaluation Agent measures coherence between the customer request and the generated reply, but it does not independently verify every statement against the internal policy and claim records. Therefore, a coherent response is not necessarily factually correct in every detail.

Other current limitations include:

- evaluation performed by a language model rather than human reviewers;
- no persistent production database;
- no authentication or customer-facing interface;
- no production monitoring or audit system;
- results may vary with the selected model and prompts.

## Possible Improvements

Future improvements could include:

- saving individual evaluation results to a dedicated CSV;
- adding a short explanation for every evaluation;
- checking generated answers directly against internal records;
- analyzing the 25 non-coherent responses by error category;
- introducing automated tests for data loading and tool execution;
- adding human-reviewed examples as an evaluation benchmark;
- comparing different prompts or models using the same dataset.

## What This Project Demonstrates

This project demonstrates how to build a simple but complete LLM-based workflow that combines:

- structured data generation;
- email classification;
- tool-assisted information retrieval;
- controlled automated decision-making;
- human escalation;
- structured output validation;
- batch processing;
- automated response evaluation.

The focus is not on replacing human insurance operators, but on automatically handling straightforward requests while routing ambiguous or sensitive cases to a person.
