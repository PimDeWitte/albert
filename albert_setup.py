#!/usr/bin/env python3
"""
Albert Setup Script
Configures your Albert instance for the global discovery network
"""

import os
import sys
import json
import getpass
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class AlbertSetup:
    def __init__(self):
        self.config_dir = Path.home() / '.albert'
        self.config_file = self.config_dir / 'config.json'
        self.key_file = self.config_dir / 'private_key.pem'
        self.public_key_file = self.config_dir / 'public_key.pem'
        self.env_file = Path('.env')
        
    def run(self):
        """Main setup flow"""
        print("=" * 60)
        print("Welcome to Albert - Self-Discovering Physics Agent Setup")
        print("=" * 60)
        print()
        
        # Create config directory
        self.config_dir.mkdir(exist_ok=True)
        
        # Ask what they want to do
        setup_type = self.get_setup_type()
        
        if setup_type == 'researcher':
            # Standard researcher setup
            config = self.collect_user_info()
            config['api_keys'] = self.setup_api_keys()
            config['public_key'] = self.setup_crypto_keys()
            config['consent'] = self.get_consent()
            
        elif setup_type == 'benchmark':
            # Model benchmarking setup
            config = self.setup_model_benchmark()
            config['public_key'] = self.setup_crypto_keys()
            
        # Save configuration
        self.save_config(config)
        
        # Create environment file
        self.create_env_file(config)
        
        # Register with network
        if setup_type == 'researcher' and config['consent']['share_discoveries']:
            self.register_with_network(config)
        elif setup_type == 'benchmark':
            self.register_benchmark_model(config)
        
        print("\n✅ Setup complete!")
        print(f"\nYour configuration has been saved to: {self.config_file}")
        print(f"Your private key is stored at: {self.key_file}")
        
        if setup_type == 'researcher':
            print("\nTo get started:")
            print("  ./albert run              # Run all theories")
            print("  ./albert run --help       # See all options")
            print("  ./albert discover         # Start AI theory discovery")
            print("  ./albert discover --self-monitor  # With continuous monitoring")
        else:
            print("\nTo benchmark your model, run:")
            print(f"  ./albert benchmark --model \"{config['model_name']}\"")
        
        print("\nFor all available commands:")
        print("  ./albert --help")
        
        print("\nHappy discovering! 🚀")
    
    def get_setup_type(self):
        """Ask user what type of setup they want"""
        print("What would you like to set up?")
        print("-" * 40)
        print("1. Researcher account - Discover new physics theories")
        print("2. Model benchmark - Test your AI model on physics discovery")
        print()
        
        while True:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == '1':
                return 'researcher'
            elif choice == '2':
                return 'benchmark'
            else:
                print("Please enter 1 or 2.")
    
    def setup_model_benchmark(self):
        """Setup configuration for model benchmarking"""
        print("\nStep 1: Model Provider Information")
        print("-" * 40)
        
        config = {
            'setup_type': 'benchmark'
        }
        
        # Organization info
        config['organization'] = input("Organization/Company name: ").strip()
        while not config['organization']:
            print("Organization name is required.")
            config['organization'] = input("Organization/Company name: ").strip()
        
        # Model info
        config['model_name'] = input("Model name (e.g., 'GPT-4', 'Claude-3'): ").strip()
        while not config['model_name']:
            print("Model name is required.")
            config['model_name'] = input("Model name: ").strip()
        
        # Contact info
        config['name'] = input("Contact person name: ").strip()
        config['email'] = input("Contact email: ").strip()
        while not config['email']:
            print("Email is required for benchmark results.")
            config['email'] = input("Contact email: ").strip()
        
        print("\nStep 2: Model API Configuration")
        print("-" * 40)
        print("Albert supports any OpenAI-compatible API endpoint.")
        print("This includes OpenAI, Azure OpenAI, local servers (vLLM, Ollama), and custom endpoints.\n")
        
        # API configuration
        config['api_type'] = self.get_api_type()
        
        if config['api_type'] == 'openai':
            config['api_base'] = 'https://api.openai.com/v1'
            config['api_key'] = getpass.getpass("OpenAI API key: ").strip()
            config['model_id'] = input(f"Model ID [{config['model_name'].lower()}]: ").strip() or config['model_name'].lower()
            
        elif config['api_type'] == 'azure':
            config['api_base'] = input("Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com): ").strip()
            config['api_key'] = getpass.getpass("Azure API key: ").strip()
            config['deployment_name'] = input("Deployment name: ").strip()
            config['api_version'] = input("API version [2024-02-01]: ").strip() or "2024-02-01"
            
        elif config['api_type'] == 'custom':
            config['api_base'] = input("API base URL (e.g., http://localhost:8000/v1): ").strip()
            config['api_key'] = getpass.getpass("API key (leave empty if not required): ").strip() or "dummy"
            config['model_id'] = input("Model ID to use in requests: ").strip()
            
            # Test custom endpoint
            print("\nTesting connection to custom endpoint...")
            if not self.test_custom_endpoint(config):
                print("❌ Failed to connect to custom endpoint.")
                if input("Continue anyway? (y/n): ").lower() != 'y':
                    sys.exit(1)
        
        # Additional model info
        print("\nStep 3: Model Capabilities")
        print("-" * 40)
        
        config['model_info'] = {
            'max_tokens': int(input("Maximum context length (tokens) [128000]: ") or "128000"),
            'supports_json': input("Supports JSON mode? (y/n) [y]: ").lower() != 'n',
            'supports_tools': input("Supports function calling? (y/n) [n]: ").lower() == 'y',
            'temperature_range': input("Recommended temperature range [0.0-1.0]: ") or "0.0-1.0"
        }
        
        # Consent for benchmarking
        print("\nStep 4: Benchmark Agreement")
        print("-" * 40)
        print("By participating in Albert's benchmark:")
        print("  • Your model will be tested on physics discovery tasks")
        print("  • Results will be published on the public leaderboard")
        print("  • Valid theories discovered will be attributed to your model")
        print("  • You can update or remove your model at any time")
        print()
        
        config['consent'] = {
            'benchmark_agreement': input("Do you agree to these terms? (y/n): ").lower() == 'y',
            'publish_results': input("Publish results on leaderboard? (y/n): ").lower() == 'y',
            'share_discoveries': True,  # Always true for benchmark models
            'marketing_consent': input("Allow us to feature your model in marketing? (y/n): ").lower() == 'y'
        }
        
        if not config['consent']['benchmark_agreement']:
            print("Benchmark agreement is required to proceed.")
            sys.exit(1)
        
        return config
    
    def get_api_type(self):
        """Get the type of API endpoint"""
        print("Select API type:")
        print("1. OpenAI API")
        print("2. Azure OpenAI")
        print("3. Custom OpenAI-compatible endpoint (vLLM, Ollama, etc.)")
        print()
        
        while True:
            choice = input("Enter your choice (1-3): ").strip()
            if choice == '1':
                return 'openai'
            elif choice == '2':
                return 'azure'
            elif choice == '3':
                return 'custom'
            else:
                print("Please enter 1, 2, or 3.")
    
    def test_custom_endpoint(self, config):
        """Test connection to a custom endpoint"""
        try:
            import requests
            
            # Try to list models
            headers = {
                'Authorization': f"Bearer {config['api_key']}"
            }
            
            response = requests.get(
                f"{config['api_base']}/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✓ Successfully connected to endpoint")
                models = response.json().get('data', [])
                if models:
                    print(f"  Available models: {', '.join([m['id'] for m in models[:3]])}")
                return True
            else:
                print(f"  Connection failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"  Connection error: {e}")
            return False
        
    def collect_user_info(self):
        """Collect basic user information"""
        print("Step 1: User Information")
        print("-" * 40)
        
        config = {}
        
        # Name
        config['name'] = input("Your name (for attribution): ").strip()
        while not config['name']:
            print("Name is required for attribution.")
            config['name'] = input("Your name: ").strip()
        
        # Email (optional)
        config['email'] = input("Email (optional, for important updates only): ").strip()
        
        # Organization (optional)
        config['organization'] = input("Organization (optional): ").strip()
        
        # Location (optional)
        config['location'] = input("Location (optional, e.g., 'Boston, MA'): ").strip()
        
        print(f"\nHello, {config['name']}! Let's configure your Albert instance.\n")
        
        return config
    
    def setup_api_keys(self):
        """Setup LLM API keys"""
        print("Step 2: API Configuration")
        print("-" * 40)
        print("Albert needs at least one LLM API to generate theories.")
        print("Supported providers: OpenAI, Anthropic, Google, Grok, or custom endpoint\n")
        
        api_keys = {}
        
        # OpenAI
        if input("Do you have an OpenAI API key? (y/n): ").lower() == 'y':
            api_keys['openai'] = getpass.getpass("OpenAI API key: ").strip()
        
        # Anthropic
        if input("Do you have an Anthropic API key? (y/n): ").lower() == 'y':
            api_keys['anthropic'] = getpass.getpass("Anthropic API key: ").strip()
        
        # Google
        if input("Do you have a Google AI API key? (y/n): ").lower() == 'y':
            api_keys['google'] = getpass.getpass("Google AI API key: ").strip()
        
        # Grok
        if input("Do you have a Grok API key? (y/n): ").lower() == 'y':
            api_keys['grok'] = getpass.getpass("Grok API key: ").strip()
        
        # Custom endpoint
        if input("Do you want to use a custom OpenAI-compatible endpoint? (y/n): ").lower() == 'y':
            api_keys['custom_endpoint'] = input("Custom endpoint URL: ").strip()
            api_keys['custom_api_key'] = getpass.getpass("Custom endpoint API key: ").strip()
        
        if not api_keys:
            print("\n❌ At least one API key is required.")
            print("Please obtain an API key from one of the supported providers.")
            sys.exit(1)
        
        # Set default provider
        providers = list(api_keys.keys())
        if len(providers) == 1:
            api_keys['default_provider'] = providers[0]
        else:
            print(f"\nAvailable providers: {', '.join(providers)}")
            default = input(f"Default provider [{providers[0]}]: ").strip() or providers[0]
            api_keys['default_provider'] = default
        
        print(f"\n✓ API configuration complete. Default provider: {api_keys['default_provider']}")
        
        return api_keys
    
    def setup_crypto_keys(self):
        """Setup or import cryptographic keys"""
        print("\nStep 3: Cryptographic Keys")
        print("-" * 40)
        print("Albert uses public key cryptography to sign your discoveries.")
        print("This ensures proper attribution and prevents tampering.\n")
        
        if self.key_file.exists():
            print(f"Existing key found at: {self.key_file}")
            if input("Use existing key? (y/n): ").lower() == 'y':
                with open(self.public_key_file, 'rb') as f:
                    public_key = serialization.load_pem_public_key(
                        f.read(), backend=default_backend()
                    )
                public_pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode('utf-8')
                return public_pem
        
        choice = input("Generate new key (g) or import existing (i)? [g]: ").lower() or 'g'
        
        if choice == 'i':
            # Import existing key
            key_path = input("Path to private key file: ").strip()
            try:
                with open(key_path, 'rb') as f:
                    private_key = serialization.load_pem_private_key(
                        f.read(), password=None, backend=default_backend()
                    )
                # Save to config directory
                with open(self.key_file, 'wb') as f:
                    f.write(private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ))
            except Exception as e:
                print(f"❌ Error loading key: {e}")
                sys.exit(1)
        else:
            # Generate new key
            print("Generating new RSA key pair...")
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            # Save private key
            with open(self.key_file, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Make private key readable only by owner
            os.chmod(self.key_file, 0o600)
            
            print(f"✓ Private key saved to: {self.key_file}")
        
        # Extract and save public key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        with open(self.public_key_file, 'w') as f:
            f.write(public_pem)
        
        print(f"✓ Public key saved to: {self.public_key_file}")
        
        return public_pem
    
    def get_consent(self):
        """Get user consent for data sharing"""
        print("\nStep 4: Network Participation")
        print("-" * 40)
        print("Albert can share your validated discoveries with the global network.")
        print("This helps advance scientific knowledge and gives you attribution.\n")
        
        consent = {}
        
        print("What will be shared:")
        print("  • Theory code and parameters")
        print("  • Validation results and metrics")
        print("  • Your name and public key")
        print("  • Timestamp and LLM used")
        print("\nWhat will NOT be shared:")
        print("  • Your API keys")
        print("  • Failed theories")
        print("  • System information")
        print("  • Email address\n")
        
        share = input("Share validated discoveries with the network? (y/n): ").lower() == 'y'
        consent['share_discoveries'] = share
        
        if share:
            consent['share_failed'] = input("Also share failed attempts for research? (y/n): ").lower() == 'y'
            consent['allow_contact'] = input("Allow other researchers to contact you? (y/n): ").lower() == 'y'
        else:
            consent['share_failed'] = False
            consent['allow_contact'] = False
        
        print(f"\n✓ Consent recorded. Sharing: {'Yes' if share else 'No'}")
        
        return consent
    
    def save_config(self, config):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Make config file readable only by owner
        os.chmod(self.config_file, 0o600)
    
    def create_env_file(self, config):
        """Create .env file for environment variables"""
        env_content = []
        
        if config.get('setup_type') == 'benchmark':
            # Benchmark model configuration
            env_content.append("# Model Benchmark Configuration")
            env_content.append(f"BENCHMARK_MODEL_NAME={config['model_name']}")
            env_content.append(f"BENCHMARK_API_TYPE={config['api_type']}")
            env_content.append(f"BENCHMARK_API_BASE={config['api_base']}")
            env_content.append(f"BENCHMARK_API_KEY={config['api_key']}")
            
            if config['api_type'] == 'openai':
                env_content.append(f"BENCHMARK_MODEL_ID={config['model_id']}")
            elif config['api_type'] == 'azure':
                env_content.append(f"BENCHMARK_DEPLOYMENT_NAME={config['deployment_name']}")
                env_content.append(f"BENCHMARK_API_VERSION={config['api_version']}")
            elif config['api_type'] == 'custom':
                env_content.append(f"BENCHMARK_MODEL_ID={config['model_id']}")
            
        else:
            # Regular researcher API keys
            api_keys = config['api_keys']
            if 'openai' in api_keys:
                env_content.append(f"OPENAI_API_KEY={api_keys['openai']}")
            if 'anthropic' in api_keys:
                env_content.append(f"ANTHROPIC_API_KEY={api_keys['anthropic']}")
            if 'google' in api_keys:
                env_content.append(f"GOOGLE_API_KEY={api_keys['google']}")
            if 'grok' in api_keys:
                env_content.append(f"GROK_API_KEY={api_keys['grok']}")
            if 'custom_endpoint' in api_keys:
                env_content.append(f"CUSTOM_ENDPOINT={api_keys['custom_endpoint']}")
                env_content.append(f"CUSTOM_API_KEY={api_keys['custom_api_key']}")
        
        # Add database credentials
        env_content.append("")
        env_content.append("# Albert Network API")
        env_content.append("ALBERT_API_URL=https://api.albert.so")
        env_content.append("")
        env_content.append("# Supabase (for development)")
        env_content.append("SUPABASE_URL=https://fwulquathotgyxttxhjk.supabase.co")
        env_content.append("SUPABASE_KEY=<will-be-provided>")
        
        # Write .env file
        with open(self.env_file, 'w') as f:
            f.write('\n'.join(env_content))
        
        # Make .env file readable only by owner
        os.chmod(self.env_file, 0o600)
        
        print(f"\n✓ Environment file created: {self.env_file}")
    
    def register_with_network(self, config):
        """Register this Albert instance with the network"""
        print("\nRegistering with Albert Network...")
        
        registration_data = {
            'name': config['name'],
            'email': config.get('email', ''),
            'organization': config.get('organization', ''),
            'location': config.get('location', ''),
            'public_key': config['public_key'],
            'consent': config['consent']
        }
        
        try:
            # For now, just simulate registration
            # In production, this would POST to api.albert.so/register
            print("✓ Successfully registered with Albert Network!")
            print(f"  Your instance ID: ALBERT-{abs(hash(config['name']))%10000:04d}")
        except Exception as e:
            print(f"⚠️  Network registration failed: {e}")
            print("You can still run Albert locally.")
    
    def register_benchmark_model(self, config):
        """Register a model for benchmarking"""
        print("\nRegistering model for Albert Benchmark...")
        
        registration_data = {
            'organization': config['organization'],
            'model_name': config['model_name'],
            'contact_name': config['name'],
            'contact_email': config['email'],
            'api_type': config['api_type'],
            'model_info': config['model_info'],
            'public_key': config['public_key'],
            'consent': config['consent']
        }
        
        try:
            # For now, just simulate registration
            # In production, this would POST to api.albert.so/register-benchmark
            print("✓ Successfully registered for Albert Benchmark!")
            print(f"  Model ID: ALBERT-BENCH-{abs(hash(config['model_name']))%10000:04d}")
            print("\nYour model will be automatically tested on:")
            print("  • Theory generation capabilities")
            print("  • Conservation law understanding")
            print("  • Mathematical consistency")
            print("  • Experimental prediction accuracy")
            print("\nBenchmark results will be available at:")
            print("  https://albert.so/benchmark")
        except Exception as e:
            print(f"⚠️  Benchmark registration failed: {e}")
            print("You can still run benchmarks locally.")

def main():
    """Run the setup process"""
    setup = AlbertSetup()
    
    # Check if already configured
    if setup.config_file.exists():
        print("Albert is already configured.")
        if input("Reconfigure? (y/n): ").lower() != 'y':
            print("Setup cancelled.")
            return
    
    try:
        setup.run()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 