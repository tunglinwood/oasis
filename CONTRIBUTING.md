🏝️ **Welcome to OASIS!** 🏝️

Thank you for your interest in contributing to the OASIS project! 🎉 We're excited to have your support. As an open-source initiative in a rapidly evolving and open-ended field, we wholeheartedly welcome contributions of all kinds. Whether you want to introduce new features, enhance the infrastructure, improve documentation, asking issues, add more examples, implement state-of-the-art research ideas, or fix bugs, we appreciate your enthusiasm and efforts. 🙌  You are welcome to join our [discord](https://discord.com/channels/1115015097560076329/1315102455624892469) or [wechat group](assets/wechatgroup.png) for more efficient communication. 💬

## Join Our Community 🌍

### Developer Meeting Time & Link 💻

- English speakers: Coming soon.
- Chinese Speakers: Thursday at 10 PM UTC+8. Join via TecentMeeting: [Meeting Link](https://meeting.tencent.com/dm/4D2TCb67tTyB)

### Our Communication Channels 💬

- **Discord:** [Join here](https://discord.com/channels/1115015097560076329/1315102455624892469)
- **WeChat:** Scan the QR code [here](assets/wechatgroup.png)

## Guidelines 📝

### Contributing to the Code 👨‍💻👩‍💻

If you're eager to contribute to this project, that's fantastic! We're thrilled to have your support.

- If you are a contributor from the community:
  - Follow the [Fork-and-Pull-Request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow when opening your pull requests.
- If you are a member of [CAMEL-AI.org](https://github.com/camel-ai) or a collaborator of OASIS:
  - Follow the [Checkout-and-Pull-Request](https://dev.to/ceceliacreates/how-to-create-a-pull-request-on-github-16h1) workflow when opening your pull request; this will allow the PR to pass all tests that require [GitHub Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets).

Make sure to mention any related issues and tag the relevant maintainers too. 💪

Before your pull request can be merged, it must pass the formatting, linting, and testing checks. You can find instructions on running these checks locally under the **Common Actions** section below. 🔍

Ensuring excellent documentation and thorough testing is absolutely crucial. Here are some guidelines to follow based on the type of contribution you're making:

- If you fix a bug:
  - Add a relevant unit test when possible. These can be found in the `test` directory.
- If you make an improvement:
  - Update any affected example console scripts in the `examples` directory, and documentation in the `docs` directory.
  - Update unit tests when relevant.
- If you add a feature:
  - Include unit tests in the `test` directory.
  - Add a demo script in the `examples` directory.

### Recent Contributions 🚀

This section documents recent significant contributions to the OASIS project:

- **CLI Enhancement for User Profile Generation**: Added command-line interface arguments to both user generation scripts (`user_generate.py` and `user_generate_detailed.py`) to support:
  - `--openai-api-key`: Your OpenAI API key (required)
  - `--openai-base-url`: Custom OpenAI base URL (optional)
  - `--model-name`: Model name to use (optional, defaults to "gpt-3.5-turbo")
  - `--max-workers`: Maximum number of worker threads (optional, defaults to 100)
  - `--N-of-agents`: Number of user profiles to generate (optional, defaults to 10000)
  - `--output-path`: Output file path (optional, defaults vary by mode)
- **Unified User Generation Script**: Created `generate_users.py` that serves as a convenient wrapper to run either simple or detailed generation with a single command
- **Comprehensive Documentation**: Created detailed documentation in `docs/user_generation_guide.md` explaining how to use both simple and detailed user generation scripts
- **Consistency Features**: Enhanced detailed generator with data consistency mechanisms including age-appropriate education levels, realistic income expectations, geographically consistent ethnicity distributions, and more

We're a small team focused on building great things. If you have something in mind that you'd like to add or modify, opening a pull request is the ideal way to catch our attention. 🚀

### Contributing to Code Reviews 🔍

This part outlines the guidelines and best practices for conducting code reviews in OASIS. The aim is to ensure that all contributions are of high quality, align with the project's goals, and are consistent with our coding standards.

#### Purpose of Code Reviews

- Maintain Code Quality: Ensure that the codebase remains clean, readable, and maintainable.
- Knowledge Sharing: Facilitate knowledge sharing among contributors and help new contributors learn best practices.
- Bug Prevention: Catch potential bugs and issues before they are merged into the main branch.
- Consistency: Ensure consistency in style, design patterns, and architecture across the project.

#### Review Process Overview

- Reviewers should check the code for functionality, readability, consistency, and compliance with the project’s coding standards.
- If changes are necessary, the reviewer should leave constructive feedback.
- The contributor addresses feedback and updates the PR.
- The reviewer re-reviews the updated code.
- Once the code is approved by at least one reviewer, it can be merged into the main branch.
- Merging should be done by a maintainer or an authorized contributor.

#### Code Review Checklist

- Functionality

  - Correctness: Does the code perform the intended task? Are edge cases handled?
  - Testing: Is there sufficient test coverage? Do all tests pass?
  - Security: Are there any security vulnerabilities introduced by the change?
  - Performance: Does the code introduce any performance regressions?

- Code Quality

  - Readability: Is the code easy to read and understand? Is it well-commented where necessary?
  - Maintainability: Is the code structured in a way that makes future changes easy?
  - Style: Does the code follow the project’s style guidelines?
    Currently we use Ruff for format check and take [Google Python Style Guide](%22https://google.github.io/styleguide/pyguide.html%22) as reference.
  - Documentation: Are public methods, classes, and any complex logic well-documented?

- Design

  - Consistency: Does the code follow established design patterns and project architecture?
  - Modularity: Are the changes modular and self-contained? Does the code avoid unnecessary duplication?
  - Dependencies: Are dependencies minimized and used appropriately?

#### Reviewer Responsibilities

- Timely Reviews: Reviewers should strive to review PRs promptly to keep the project moving.
- Constructive Feedback: Provide feedback that is clear, constructive, and aimed at helping the contributor improve.
- Collaboration: Work with the contributor to address any issues and ensure the final code meets the project’s standards.
- Approvals: Only approve code that you are confident meets all the necessary criteria.

#### Common Pitfalls

- Large PRs: Avoid submitting PRs that are too large. Break down your changes into smaller, manageable PRs if possible.
- Ignoring Feedback: Address all feedback provided by reviewers, even if you don’t agree with it—discuss it instead of ignoring it.
- Rushed Reviews: Avoid rushing through reviews. Taking the time to thoroughly review code is critical to maintaining quality.

Code reviews are an essential part of maintaining the quality and integrity of our open source project. By following these guidelines, we can ensure that OASIS remains robust, secure, and easy to maintain, while also fostering a collaborative and welcoming community.

### Guideline for Writing Docstrings

This guideline will help you write clear, concise, and structured docstrings for contributing to `OASIS`.

#### 1. Use the Triple-Quoted String with `r"""` (Raw String)

Begin the docstring with `r"""` to indicate a raw docstring. This prevents any issues with special characters and ensures consistent formatting.

#### 2. Provide a Brief Class or Method Description

- Start with a concise summary of the purpose and functionality.
- Keep each line under `79` characters.
- The summary should start on the first line without a linebreak.

Example:

```python
r"""Class for managing conversations of OASIS Agents.
"""
```

#### 3. Document Parameters in the Args Section

- Use an `Args`: section for documenting constructor or function parameters.
- Maintain the `79`-character limit for each line, and indent continuation lines by 4 spaces.
- Follow this structure:
  - Parameter Name: Match the function signature.
  - Type: Include the type (e.g., `int`, `str`, custom types like `BaseModelBackend`).
  - Description: Provide a brief explanation of the parameter's role.
  - Default Value: Use (`default: :obj:<default_value>`) to indicate default values.

Example:

```markdown
Args:
    system_message (BaseMessage): The system message for initializing
        the agent's conversation context.
    model (BaseModelBackend, optional): The model backend to use for
        response generation. Defaults to :obj:`OpenAIModel` with
        `GPT_4O_MINI`. (default: :obj:`OpenAIModel` with `GPT_4O_MINI`)
```

### Principles 🛡️

#### Naming Principle: Avoid Abbreviations in Naming

- Abbreviations can lead to ambiguity, especially since variable names and code in OASIS are directly used by agents.
- Use clear, descriptive names that convey meaning without requiring additional explanation. This improves both human readability and the agent's ability to interpret the code.

Examples:

- Bad: msg_win_sz
- Good: message_window_size

By adhering to this principle, we ensure that OASIS remains accessible and unambiguous for both developers and AI agents.

#### Logging Principle: Use `logger` Instead of `print`

Avoid using `print` for output. Use Python's `logging` module (`logger`) to ensure consistent, configurable, and professional logging.

Examples:

- Bad:
  ```python
  print("Process started")
  print(f"User input: {user_input}")
  ```
- Good:
  ```python
  Args:
  logger.info("Process started")
  logger.debug(f"User input: {user_input}")
  ```

### Board Item Create Workflow 🛠️

At OASIS, we manage our project through a structured workflow that ensures efficiency and clarity in our development process. Our workflow includes stages for issue creation and pull requests (PRs), sprint planning, and reviews.

#### Issue Item Stage:

Our [issues](https://github.com/camel-ai/oasis/issues) page on GitHub is regularly updated with bugs, improvements, and feature requests. We have a handy set of labels to help you sort through and find issues that interest you. Feel free to use these labels to keep things organized.

When you start working on an issue, please assign it to yourself so that others know it's being taken care of. If you're unable to assign it to yourself because you're not an OASIS collaborator, feel free to leave a comment on the issue instead.

When creating a new issue, it's best to keep it focused on a specific bug, improvement, or feature. If two issues are related or blocking each other, it's better to link them instead of merging them into one.

We do our best to keep these issues up to date, but considering the fast-paced nature of this field, some may become outdated. If you come across any such issues, please give us a heads-up so we can address them promptly. 👀

Here’s how to engage with our issues effectively:

- Go to [GitHub Issues](https://github.com/camel-ai/oasis/issues), create a new issue, choose the category, and fill in the required information.
- Ensure the issue has a proper title and update the Assignees, Labels, Projects (select Backlog status), Development, and Milestones.
- Discuss the issue during team meetings, then move it to the Analysis Done column.
- At the beginning of each sprint, share the analyzed issue and move it to the Sprint Planned column if you are going to work on this issue in the sprint.

#### Pull Request Item Stage:

- Go to [GitHub Pulls](https://github.com/camel-ai/oasis/pulls), create a new PR, choose the branch, and fill in the information, linking the related issue.
- Ensure the PR has a proper title and update the Reviewers (convert to draft), Assignees, Labels, Projects (select Developing status), Development, and Milestones.
- If the PR is related to a roadmap, link the roadmap to the PR.
- Move the PR item through the stages: Developing, Stuck, Reviewing (click ready for review), Merged. The linked issue will close automatically when the PR is merged.

**Labeling PRs:**

- **feat**: For new features (e.g., `feat: Add new AI model`)
- **fix**: For bug fixes (e.g., `fix: Resolve memory leak issue`)
- **docs**: For documentation updates (e.g., `docs: Update contribution guidelines`)
- **style**: For code style changes (e.g., `style: Refactor code formatting`)
- **refactor**: For code refactoring (e.g., `refactor: Optimize data processing`)
- **test**: For adding or updating tests (e.g., `test: Add unit tests for new feature`)
- **chore**: For maintenance tasks (e.g., `chore: Update dependencies`)

### Sprint Planning & Review 🎯

#### Definition

Sprint planning defines what can be delivered in the sprint and how it will be achieved. Sprint review allows stakeholders to review and provide feedback on recent work.

#### Practice

- **Sprint Duration**: Four weeks for development and review.
- **Sprint Planning & Review**: Conducted biweekly during the dev meeting (around 30 minutes).
- **Planning**: Founder highlights the sprint goal and key points; developers pick items for the sprint.
- **Review**: Feedback on delivered features and identification of improvement areas.

### Getting Help 🆘

Our aim is to make the developer setup as straightforward as possible. If you encounter any challenges during the setup process, don't hesitate to reach out to a maintainer. We're here to assist you and ensure that the experience is smooth not just for you but also for future contributors. 😊

In line with this, we do have specific guidelines for code linting, formatting, and documentation in the codebase. If you find these requirements difficult or even just bothersome to work with, please feel free to get in touch with a maintainer — you can *@doudou_wu in Discord or @张再斌 in the WeChat group*. We don't want these guidelines to hinder the integration of good code into the codebase, so we're more than happy to provide support and find a solution that works for you. 🤝

## Quick Start 🚀

To get started with OASIS, follow these steps:

```sh
# Clone github repo
git clone https://github.com/camel-ai/oasis.git

# Change directory into project directory
cd oasis

# Activate oasis virtual environment
poetry shell

# Install oasis from source
poetry install

# The following command installs a pre-commit hook into the local git repo,
# so every commit gets auto-formatted and linted.
pre-commit install

# Run oasis's pre-commit before push
pre-commit run --all-files

# Run oasis's unit tests
pytest test

# Exit the virtual environment
exit
```

These commands will install all the necessary dependencies for running the package, examples, linting, formatting, tests, and coverage.

To verify that everything is set up correctly, run `pytest .` This will ensure that all tests pass successfully. ✅

> \[!TIP\]
> You need to config OPENAI API Keys as environment variables to pass all tests.

## Generating User Profiles with CLI Arguments 🧑‍💼

OASIS now supports command-line interface (CLI) arguments for generating user profiles in both simple and detailed modes. The following arguments are available:

### User Profile Generation Scripts

Two scripts are available for generating synthetic user profiles:

1. **Simple Profile Generator** (`generator/reddit/user_generate.py`): Creates basic user profiles with fundamental demographic and personality information
2. **Detailed Profile Generator** (`generator/reddit/user_generate_detailed.py`): Creates comprehensive user profiles with extensive demographic, socioeconomic, and psychological attributes
3. **Unified Generator** (`generator/reddit/generate_users.py`): A convenient wrapper to run either simple or detailed generation with a single command

### CLI Arguments

The scripts support the following arguments:

- `--openai-api-key`: Your OpenAI API key (required)
- `--openai-base-url`: Custom OpenAI base URL (optional)
- `--model-name`: Model name to use (optional, defaults to "gpt-3.5-turbo")
- `--max-workers`: Maximum number of worker threads (optional, defaults to 100)
- `--N-of-agents`: Number of user profiles to generate (optional, defaults to 10000)
- `--output-path`: Output file path (optional, defaults vary by mode)

### Usage Examples

```bash
# Generate simple profiles with specific parameters
python generator/reddit/user_generate.py \
  --openai-api-key sk-your-api-key-here \
  --model-name gpt-4 \
  --N-of-agents 5000 \
  --output-path ./results/simple_users_5000.json \
  --max-workers 50

# Generate detailed profiles
python generator/reddit/user_generate_detailed.py \
  --openai-api-key sk-your-api-key-here \
  --model-name gpt-4 \
  --N-of-agents 3000 \
  --output-path ./results/detailed_users_3000.json \
  --max-workers 75

# Use the unified generator (recommended)
# Generate simple profiles
python generator/reddit/generate_users.py simple \
  --openai-api-key sk-your-api-key-here \
  --N-of-agents 2000 \
  --model-name gpt-4

# Generate detailed profiles
python generator/reddit/generate_users.py detailed \
  --openai-api-key sk-your-api-key-here \
  --N-of-agents 1000 \
  --output-path ./results/my_detailed_users.json \
  --max-workers 50
```

For comprehensive information on all available options and generated data attributes, see the user generation guide in `docs/user_generation_guide.md`.

## Common Actions 🔄

### Update dependencies

Whenever you add, update, or delete any dependencies in `pyproject.toml`, please run `poetry lock` to synchronize the dependencies with the lock file.

### Coverage 📊

Code coverage measures the extent to which unit tests cover the code, helping identify both robust and less robust areas of the codebase.

To generate a report showing the current code coverage, execute one of the following commands.

To include all source files into coverage:

```bash
coverage erase
coverage run --source=. -m pytest .
coverage html
# Open htmlcov/index.html
```

To include only tested files:

```bash
pytest --cov --cov-report=html
```

The coverage report will be generated at `htmlcov/index.html`.

### Tests 🧪

Currently, the test setup requires an OpenAI API key to test the framework, making them resemble integration tests.

- For Bash shell (Linux, macOS, Git Bash on Windows):\*\*

```bash
# Export your OpenAI API key
export OPENAI_API_KEY=<insert your OpenAI API key>
```

- For Windows Command Prompt:\*\*

```cmd
REM export your OpenAI API key
set OPENAI_API_KEY=<insert your OpenAI API key>
```

To run all tests including those that use OpenAI API, use the following command:

```bash
pytest .
```

## Documentation 📚

### Contribute to Documentation 📝

We use [Mintlify](https://mintlify.com/) for documentation.

We kindly request that you provide comprehensive documentation for all classes and methods to ensure high-quality documentation coverage.

### Build Documentation Locally 🛠️

To build the documentation locally, follow these steps:

1. Install the Mintlify CLI:

   ```sh
   npm install -g mintlify
   ```

1. Navigate to `docs` Directory:

   ```sh
   cd docs
   ```

1. Run the Mintlify development server:

   ```sh
   mintlify dev
   ```

   This will start a local server where you can preview your changes.

More guidelines about building and hosting documentations locally can be found [here](https://github.com/camel-ai/oasis/tree/main/docs/README.md).

## Versioning and Release 🚀

As of now, OASIS is actively under development and the latest version has been published to PyPI.

OASIS follows the [semver](https://semver.org/) versioning standard. As pre-1.0 software, even patch releases may contain [non-backwards-compatible changes](https://semver.org/#spec-item-4). Currently, the major version is 0, and the minor version is incremented. Releases are made once the maintainers feel that a significant body of changes has accumulated.

## License 📜

The source code of the OASIS project is licensed under Apache 2.0. Your contributed code will be also licensed under Apache 2.0 by default. To add license to you code, you can manually copy-paste it from `license_template.txt` to the head of your files or run the `update_license.py` script to automate the process:

```bash
python licenses/update_license.py . licenses/license_template.txt
```

This script will add licenses to all the `*.py` files or update the licenses if the existing licenses are not the same as `license_template.txt`.

## Giving Credit 🎉

If your contribution has been included in a release, we'd love to give you credit on Twitter, Reddit, or Rednote (小红书)—but only if you're comfortable with it!

If you have accounts on any of these platforms that you would like us to mention, please let us know either in the pull request or through another communication method. We want to make sure you receive proper recognition for your valuable contributions. 😄
