# Gravity Theory Explorer Web UI

A simple web interface for exploring, modifying, and evaluating gravitational theories.

## Features

- **Theory Browser**: View all available gravitational theories with their categories and rankings
- **Theory Modification**: Use natural language to modify existing theories (e.g., "make beta 0.75")
- **Automated Evaluation**: Queue and run modified theories through the full evaluation pipeline
- **Live Leaderboard**: See theories ranked by minimum loss with validation status
- **Result Visualization**: View trajectory plots and detailed loss breakdowns
- **Albert Network Integration**: Submit validated theories to the global discovery network
- **Global Leaderboard**: View and compare theories from researchers worldwide

## Setup

1. Install dependencies:
```bash
pip install -r physics_agent/ui/requirements.txt
```

2. Set your API key:

### Primary Provider (Recommended)
```bash
export GROK_API_KEY="your-xai-grok-api-key"
```
Get your xAI/Grok API key from: https://x.ai/api

### Experimental Providers
Other providers are experimental and may not work correctly:
```bash
export OPENAI_API_KEY="your-openai-key"        # Experimental
export ANTHROPIC_API_KEY="your-anthropic-key"  # Experimental 
export GOOGLE_API_KEY="your-google-key"        # Experimental
```

Without an API key, the system will use mock responses for testing.

3. (Optional) Configure Albert Network for global theory sharing:
```bash
# Install Albert dependencies
pip install supabase cryptography

# Run setup
python albert_setup.py
```

4. Run the server:
```bash
python -m physics_agent.ui.server
```

5. Open your browser to: http://localhost:8000

## Usage

### Browsing Theories
- Click on any theory in the list to view its source code
- Baseline theories are highlighted in blue
- Theory rankings from the latest run are shown

### Modifying Theories
1. Click on a theory to open its details
2. Enter a modification instruction, for example:
   - "make beta 0.75"
   - "add quantum correction term"
   - "increase the coupling constant by factor of 2"
   - "add logarithmic term to the metric"
3. Click "Generate Modified Theory" to create a variation
4. The system will automatically evaluate the new theory

### Viewing Results
- Check the leaderboard at `/leaderboard` to see all theories ranked by performance
- Click on any theory in the leaderboard to see:
  - Full loss breakdown across all metrics and baselines
  - Validation status
  - Trajectory visualizations

### Albert Network
- Visit `/albert` to see the global discovery network status
- Successfully validated theories are automatically submitted to the network
- View global leaderboard with filters for different time periods
- Manual submit option available for validated theories

## API Endpoints

### Theory Management
- `GET /api/theories` - List all available theories
- `GET /api/theory/<key>` - Get theory source code
- `POST /api/modify_theory` - Generate a modified theory using LLM
- `POST /api/evaluate_theory` - Queue a theory for evaluation
- `GET /api/job_status/<job_id>` - Check evaluation status

### Leaderboards
- `GET /api/leaderboard` - Get current local leaderboard data
- `GET /api/theory_details/<name>` - Get detailed results for a theory

### Albert Network
- `GET /api/albert/status` - Check Albert Network configuration and status
- `GET /api/albert/leaderboard` - Get global leaderboard (params: timeframe=all|month|week|today)
- `POST /api/albert/submit` - Submit a theory to the Albert Network

## Notes

- Theory evaluations run with reduced steps (5000) for faster results
- Custom theories are temporarily added to the theories directory during evaluation
- The leaderboard auto-refreshes every 30 seconds
- Without a Grok API key, the system will return a mock quantum-corrected theory
- Albert Network submissions require prior setup via `albert_setup.py` 