# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Lint/Test Commands
- **Run training**: `python train.py`
- **Single test run**: `python -c "from env.flag_frenzy_env import FlagFrenzyEnv; env = FlagFrenzyEnv(); obs, info = env.reset(); print(info)"`
- **Install dependencies**: `pip install -r requirements.txt`

## Code Style Guidelines
- **Imports**: Group standard libraries first, then third-party, then local imports
- **Formatting**: Use 4-space indentation; 100 character line length limit
- **Types**: Use type hints where applicable; avoid dynamic typing
- **Naming**: 
  - Classes: PascalCase (e.g., `FlagFrenzyEnv`, `CustomModel`)
  - Functions/Methods: snake_case (e.g., `get_valid_engage_mask`, `execute_action`) 
  - Constants: UPPER_SNAKE_CASE (e.g., `MAX_ENTITIES`)
- **Error Handling**: Use try/except for expected errors; log exceptions with detailed context
- **PyTorch Models**: Follow RLlib's model conventions with proper overrides
- **Environment Logic**: Update observations, rewards, and simulation state together