# Installing Albert

Albert is a self-discovering physics agent for gravitational theory research. It uses AI to generate and validate novel theories of gravity.

## Quick Install (Recommended)

Install Albert with a single command:

```bash
curl -fsSL https://github.com/pimdewitte/albert/releases/download/stable/download_cli.sh | bash
```

This will:
- Download the latest version of Albert
- Install it to `~/.albert`
- Set up a Python virtual environment
- Install all dependencies
- Create an `albert` command in your PATH
- Run the initial configuration wizard

### Non-interactive Installation

To install without the configuration wizard:

```bash
curl -fsSL https://github.com/pimdewitte/albert/releases/download/stable/download_cli.sh | CONFIGURE=false bash
```

You can run `albert setup` later to configure API keys and network settings.

## Manual Installation

If you prefer to install manually:

1. Clone the repository:
```bash
git clone https://github.com/pimdewitte/albert.git
cd albert
```

2. Run the setup script:
```bash
./setup_unified.sh
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate
```

4. Run the configuration:
```bash
python albert_setup.py
```

## System Requirements

- **OS**: macOS, Linux (Windows via WSL)
- **Python**: 3.9 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **GPU**: Optional but recommended for physics simulations
- **Storage**: ~2GB for installation and dependencies

## Configuration

Albert uses AI to generate new theories of gravity. Supported API providers:

### Primary Support
- **xAI/Grok** (recommended) - Get your API key from: https://x.ai/api

### Experimental Support
- OpenAI (GPT-4)
- Anthropic (Claude)
- Google AI (Gemini)
- Custom OpenAI-compatible endpoints

Run `albert setup` to configure your API keys and preferences.

## Updating Albert

To update to the latest version:

```bash
albert update
```

This will download and install the latest version while preserving your configuration.

## Uninstalling

To uninstall Albert:

```bash
rm -rf ~/.albert
rm ~/.local/bin/albert
```

Also remove the PATH addition from your shell configuration file (`~/.bashrc` or `~/.zshrc`).

## Troubleshooting

### Command not found

If you get "command not found" after installation, add the bin directory to your PATH:

```bash
export PATH="$PATH:$HOME/.local/bin"
```

Add this line to your `~/.bashrc` or `~/.zshrc` to make it permanent.

### Python version issues

Albert requires Python 3.9+. Check your version:

```bash
python3 --version
```

On macOS, you can install a newer Python using Homebrew:

```bash
brew install python@3.11
```

### Permission denied

If you get permission errors, make sure the installer script is executable:

```bash
chmod +x download_cli.sh
```

## Next Steps

After installation:

1. Run `albert help` to see available commands
2. Configure your API keys: `albert setup`
3. Start discovering: `albert discover --initial "unified field theory"`
4. Visualize results: `albert visualize runs/your_run_id`

For more information, see the [documentation](https://github.com/pimdewitte/albert/wiki). 