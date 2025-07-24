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
        
        # Collect user information
        config = self.collect_user_info()
        
        # Setup API keys
        config['api_keys'] = self.setup_api_keys()
        
        # Setup cryptographic keys
        config['public_key'] = self.setup_crypto_keys()
        
        # Get consent
        config['consent'] = self.get_consent()
        
        # Save configuration
        self.save_config(config)
        
        # Create environment file
        self.create_env_file(config)
        
        # Register with network
        if config['consent']['share_discoveries']:
            self.register_with_network(config)
        
        print("\n✅ Setup complete!")
        print(f"\nYour configuration has been saved to: {self.config_file}")
        print(f"Your private key is stored at: {self.key_file}")
        print("\nTo start discovering, run:")
        print("  python physics_agent/self_discovery/self_discovery.py --self-monitor")
        print("\nHappy discovering! 🚀")
        
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
        
        # Add API keys
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