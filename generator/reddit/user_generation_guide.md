# Reddit User Profile Generation Scripts Documentation

This documentation explains how to use the two Reddit user profile generation scripts that create synthetic user data for research and simulation purposes.

## Overview

The project includes two scripts for generating synthetic Reddit user profiles:

- **Simple Profile Generator** (`user_generate.py`): Creates basic user profiles with fundamental demographic and personality information
- **Detailed Profile Generator** (`user_generate_detailed.py`): Creates comprehensive user profiles with extensive demographic, socioeconomic, and psychological attributes
- **Unified Generator** (`generate_users.py`): A convenient wrapper to run either simple or detailed generation with a single command

## Prerequisites

- Python 3.7+
- OpenAI API key
- Required Python packages:
  - `openai`
  - `json`
  - `random`
  - `argparse`
  - `concurrent.futures`

## Installation

1. Install required packages:
   ```bash
   pip install openai
   ```

2. Ensure you have your OpenAI API key ready

## Script Usage

### 1. Simple User Profile Generator

#### Command Syntax
```bash
python user_generate.py --openai-api-key <API_KEY> [OPTIONS]
```

#### Required Arguments
- `--openai-api-key`: Your OpenAI API key (required)

#### Optional Arguments
- `--openai-base-url`: Custom OpenAI base URL (optional, defaults to OpenAI's official endpoint)
- `--model-name`: Model name to use (optional, defaults to "gpt-3.5-turbo")
- `--max-workers`: Maximum number of worker threads (optional, defaults to 100)
- `--N-of-agents`: Number of user profiles to generate (optional, defaults to 10000)
- `--output-path`: Output file path (optional, defaults to "./experiment_dataset/user_data/user_data_10000.json")

#### Example Usage
```bash
python user_generate.py \
  --openai-api-key sk-your-api-key-here \
  --model-name gpt-4 \
  --N-of-agents 5000 \
  --output-path ./results/simple_users_5000.json \
  --max-workers 50
```

#### Generated Data Attributes
- `realname`: Generated real name for the user
- `username`: Suggested username
- `bio`: User's bio/description
- `persona`: Detailed persona description
- `age`: Age of the user
- `gender`: Gender of the user
- `mbti`: MBTI personality type
- `country`: Country of origin
- `profession`: Profession/occupation
- `interested_topics`: List of topics the user is interested in

### 2. Detailed User Profile Generator

#### Command Syntax
```bash
python user_generate_detailed.py --openai-api-key <API_KEY> [OPTIONS]
```

#### Required Arguments
- `--openai-api-key`: Your OpenAI API key (required)

#### Optional Arguments
- `--openai-base-url`: Custom OpenAI base URL (optional, defaults to OpenAI's official endpoint)
- `--model-name`: Model name to use (optional, defaults to "gpt-3.5-turbo")
- `--max-workers`: Maximum number of worker threads (optional, defaults to 100)
- `--N-of-agents`: Number of user profiles to generate (optional, defaults to 10000)
- `--output-path`: Output file path (optional, defaults to "./experiment_dataset/user_data/generated_user_data_detailed_10000.json")

#### Example Usage
```bash
python user_generate_detailed.py \
  --openai-api-key sk-your-api-key-here \
  --model-name gpt-4 \
  --N-of-agents 3000 \
  --output-path ./results/detailed_users_3000.json \
  --max-workers 75
```

#### Generated Data Attributes
All attributes from simple generator plus:
- `education`: Education level (elementary, high school, bachelor, master, phd)
- `income`: Income bracket (low, lower-middle, middle, upper-middle, high)
- `location`: Location type (urban, suburban, rural)
- `family_status`: Family status (single, married-no-kids, etc.)
- `religion`: Religious affiliation
- `big_five_traits`: Big Five personality scores (openness, conscientiousness, extraversion, agreeableness, neuroticism)
- `sexual_orientation`: Sexual orientation
- `gender_identity`: Gender identity (cisgender, transgender, non-binary)
- `ethnicity`: Ethnicity/racial background
- `disability_status`: Disability status
- `digital_literacy`: Digital literacy level (low, medium, high, very high)
- `consumption_pattern`: Consumption patterns (essential-focused, balanced, luxury-oriented, value-conscious)
- `info_source`: Information source preferences (social media, traditional media, etc.)
- `ses_score`: Socioeconomic Status score (0-4 scale)
- `risk_tolerance`: Risk tolerance score (0-100)
- `conformity_score`: Conformity to social expectations (0-100)
- `social_trust_score`: Trust in institutions and others (0-100)
- `political_leaning`: Political leaning (0-100, lower=more liberal, higher=more conservative)

### 3. Unified User Profile Generator (Recommended)

#### Command Syntax
```bash
python generate_users.py {simple|detailed} --openai-api-key <API_KEY> [OPTIONS]
```

#### Positional Arguments
- `{simple|detailed}`: Choose between simple or detailed user profile generation

#### Required Arguments
- `--openai-api-key`: Your OpenAI API key (required)

#### Optional Arguments
- `--openai-base-url`: Custom OpenAI base URL (optional)
- `--model-name`: Model name to use (optional, defaults to "gpt-3.5-turbo")
- `--max-workers`: Maximum number of worker threads (optional, defaults to 100)
- `--N-of-agents`: Number of user profiles to generate (optional, defaults to 10000)
- `--output-path`: Output file path (optional, defaults vary by mode)

#### Example Usage
```bash
# Generate simple profiles
python generate_users.py simple \
  --openai-api-key sk-your-api-key-here \
  --N-of-agents 2000 \
  --model-name gpt-4

# Generate detailed profiles
python generate_users.py detailed \
  --openai-api-key sk-your-api-key-here \
  --N-of-agents 1000 \
  --output-path ./results/my_detailed_users.json \
  --max-workers 50
```

## Data Consistency Features

The detailed generator includes several consistency mechanisms:
- Age-appropriate education levels
- Realistic income expectations based on age and education
- Geographically consistent ethnicity distributions
- Education-appropriate profession assignments
- Age-appropriate family status
- Digital literacy adjusted based on age and education
- Information source preferences based on digital literacy and age

## Output Format

Both scripts generate JSON files containing arrays of user profile objects. Each user profile includes multiple attributes depending on the generation mode selected.

## Performance Considerations

- API costs: Each profile generation requires multiple API calls to OpenAI
- Generation time: Depends on the number of agents and max workers
- Memory usage: Larger datasets require more RAM
- Threading: The `--max-workers` parameter controls the number of concurrent API requests

## Error Handling

The scripts include error handling for API failures and will retry profile generation. The number of failures is tracked and reported at the end of the process.

## Output Directory

The scripts will create the necessary output directories if they don't exist. Generated files are saved in the experiment_dataset directory by default.