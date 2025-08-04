#!/bin/bash
# Albert Installer Script
# Self-Discovering Physics Agent for Gravitational Theory Research
# <reason>chain: Professional installer for Albert with graceful error handling and interactive/non-interactive modes</reason>

set -e  # Exit on error

# Configuration
# You can override these with your own repository if forked
REPO_URL="${REPO_URL:-https://github.com/pimdewitte/albert}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.albert}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
CONFIGURE="${CONFIGURE:-true}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helper functions
print_banner() {
    # Clear screen for better presentation
    clear
    
    # Print detailed Einstein ASCII art
    echo -e "${BLUE}"
    cat << 'EOF'
                    .+~                :xx++::
                   :`. -          .!!X!~"?!`~!~!. :-:.
                  {             .!!!H":.~ ::+!~~!!!~ `%X.
                  '             ~~!M!!>!!X?!!!!!!!!!!...!~.
                              {!:!MM!~:XM!!!!!!.:!..~ !.  `{
                  {: `   :~ .:{~!!M!XXHM!!!X!XXHtMMHHHX!  ~ ~
                ~~~~{' ~!!!:!!!!!XM!!M!!!XHMMMRMSXXX!!!!!!:  {`
                  `{  {::!!!!!X!X?M!!M!!XMMMMXXMMMM??!!!!!?!:~{
               : '~~~{!!!XMMH!!XMXMXHHXXXXM!!!!MMMMSXXXX!!!!!!!~
            :    ::`~!!!MMMMXXXtMMMMMMMMMMMHX!!!!!!HMMMMMX!!!!!: ~
               '~:~!!!!!MMMMMMMMMMMMMMMMMMMMMMXXX!!!M??MMMM!!X!!i:
               {~{!!!!!XMMMMMMMMMMMM8M8MMMMM8MMMMMXX!!!!!!!!X!?t?!:
               ~:~~!!!!?MMMMMM@M@RMRRR$@@MMRMRMMMMMMXSX!!!XMMMX{?X!
             :XX {!!XHMMMM88MM88BR$M$$$$8@8RN88MMMMMMMMHXX?MMMMMX!!!
           .:X! {XMSM8M@@$$$$$$$$$$$$$$$$$$$B8R$8MMMMMMMMMMMMMMMMX!X
          :!?! !?XMMMMM8$$$$8$$$$$$$$$$$$$$BBR$$MMM@MMMMMMMMMMMMMM!!X
        ~{!!~ {!!XMMMB$$$$$$$$$$$$$$$$$$$$$$$$MMR$8MR$MMMMMMMMMMMMM!?!:
        :~~~ !:X!XMM8$$$$$$$$$$$$$$$$$$$$$$$RR$$MMMMR8NMMMMMMMMMMMMM{!`-
    ~:{!:~`~':!:HMM8N$$$$$$$$$$$$$$$$$$$$$$$$$8MRMM8R$MRMMMMMMMMRMMMX!
  !X!``~~   :~XM?SMM$B$$$$$$$$$$$$$$$$$$$$$$BR$$MMM$@R$M$MMMMMM$MMMMX?L
 X~.      : `!!!MM#$RR$$$$$$$$$$$$$$$$$R$$$$$R$M$MMRRRM8MMMMMMM$$MMMM!?:
 ! ~ {~  !! !!~`` :!!MR$$$$$$$$$$RMM!?!??RR?#R8$M$MMMRM$RMMMM8MM$MMM!M!:>
: ' >!~ '!!  !   .!XMM8$$$$$@$$$R888HMM!!XXHWX$8$RM$MR5$8MMMMR$$@MMM!!!{ ~
!  ' !  ~!! :!:XXHXMMMR$$$$$$$$$$$$$$$$8$$$$8$$$MMR$M$$$MMMMMM$$$MMM!!!!
 ~{!!!  !!! !!HMMMMMMMM$$$$$$$$$$$$$$$$$$$$$$$$$$MMM$M$$MM8MMMR$$MMXX!!!!/:`
  ~!!!  !!! !XMMMMMMMMMMR$$$$$$$$$$$$R$RRR$$$$$$$MMMM$RM$MM8MM$$$M8MMMX!!!!:
  !~ ~  !!~ XMMM%!!!XMMX?M$$$$$$$$B$MMSXXXH?MR$$8MMMM$$@$8$M$B$$$$B$MMMX!!!!
  ~!    !! 'XMM?~~!!!MMMX!M$$$$$$MRMMM?!%MMMH!R$MMMMMM$$$MM$8$$$$$$MR@M!!!!!
  {>    !!  !Mf x@#"~!t?M~!$$$$$RMMM?Xb@!~`??MS$M@MMM@RMRMMM$$$$$$RMMMMM!!!!
  !    '!~ {!!:!?M   !@!M{XM$$R5M$8MMM$! -XXXMMRMBMMM$RMMM@$R$BR$MMMMX??!X!!
  !    '!  !!X!!!?::xH!HM:MM$RM8M$RHMMMX...XMMMMM$RMMRRMMMMMMM8MMMMMMMMX!!X!
  !     ~  !!?:::!!!MXMR~!MMMRMM8MMMMMS!!M?XXMMMMM$$M$M$RMMMM8$RMMMMMMMM%X!!
  ~     ~  !~~X!!XHMMM?~ XM$MMMMRMMMMMM@MMMMMMMMMM$8@MMMMMMMMRMMMMM?!MMM%HX!
           !!!!XSMMXXMM .MMMMMMMM$$$BB8MMM@MMMMMMMR$RMMMMMMMMMMMMMMMXX!?H!XX
           XHXMMMMMMMM!.XMMMMMMMMMR$$$8M$$$$$M@88MMMMMMMMMMMMMMM!XMMMXX!!!XM
      ~   {!MMMMMMMMRM:XMMMMMMMMMM8R$$$$$$$$$$$$$$$NMMMMMMMM?!MM!M8MXX!!/t!M
      '   ~HMMMMMMMMM~!MM8@8MMM!MM$$8$$$$$$$$$$$$$$8MMMMMMM!!XMMMM$8MR!MX!MM
          'MMMMMMMMMM'MM$$$$$MMXMXM$$$$$$$$$$$$$$$$RMMMMMMM!!MMM$$$$MMMMM{!M
          'MMMMMMMMM!'MM$$$$$RMMMMMM$$$$$$$$$$$$$$$MMM!MMMX!!MM$$$$$M$$M$M!M
           !MMMMMM$M! !MR$$$RMM8$8MXM8$$$$$$$$$$$$NMMM!MMM!!!?MRR$$RXM$$MR!M
           !M?XMM$$M.{ !MMMMMMSUSRMXM$8R$$$$$$$$$$#$MM!MMM!X!t8$M$MMMHMRMMX$
    ,-,   '!!!MM$RMSMX:.?!XMHRR$RM88$$$8M$$$$$R$$$$8MM!MMXMH!M$$RMMMMRNMMX!$
   -'`    '!!!MMMMMMMMMM8$RMM8MBMRRMR8RMMM$$$$8$8$$$MMXMMMMM!MR$MM!M?MMMMMM$
          'XX!MMMMMMM@RMM$MM@$$BM$$$M8MMMMR$$$$@$$$$MM!MMMMXX$MRM!XH!!??XMMM
          `!!!M?MHMMM$RMMMR@$$$$MR@MMMM8MMMM$$$$$$$WMM!MMMM!M$RMM!!.MM!%M?~!
           !!!!!!MMMMBMM$$RRMMMR8MMMMMRMMMMM8$$$$$$$MM?MMMM!f#RM~    `~!!!~!
           ~!!HX!!~!?MM?MMM??MM?MMMMMMMMMRMMMM$$$$$MMM!MMMM!!
           '!!!MX!:`~~`~~!~~!!!!XM!!!?!?MMMM8$$$$$MMMMXMMM!!
            !!~M@MX.. {!!X!!!!XHMHX!!``!XMMMB$MM$$B$M!MMM!!
            !!!?MRMM!:!XHMHMMMMMMMM!  X!SMMX$$MM$$$RMXMMM~
             !M!MMMM>!XMMMMMMMMXMM!!:!MM$MMMBRM$$$$8MMMM~
             `?H!M$R>'MMMM?MMM!MM6!X!XM$$$MM$MM$$$$MX$f
              `MXM$8X MMMMMMM!!MM!!!!XM$$$MM$MM$$$RX@"
               ~M?$MM !MMMMXM!!MM!!!XMMM$$$8$XM$$RM!`
                !XMMM !MMMMXX!XM!!!HMMMM$$$$RH$$M!~
                'M?MM `?MMXMM!XM!XMMMMM$$$$$RM$$#
                 `>MMk ~MMHM!XM!XMMM$$$$$$BRM$M"
                  ~`?M. !M?MXM!X$$@M$$$$$$RMM#
                    `!M  !!MM!X8$$$RM$$$$MM#`
                      !% `~~~X8$$$$8M$$RR#`
                       !!x:xH$$$$$$$R$R*`
                        ~!?MMMMRRRM@M#`       
                         `~???MMM?M"`
                             ``~~
EOF
    echo -e "${NC}"
    

}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Check system requirements
check_requirements() {
    local errors=0
    
    print_info "Checking system requirements..."
    
    # Check for curl
    if ! command -v curl &> /dev/null; then
        print_error "curl is required but not installed"
        errors=$((errors + 1))
    fi
    
    # Check for git
    if ! command -v git &> /dev/null; then
        print_error "git is required but not installed"
        errors=$((errors + 1))
    fi
    
    # Check for Python 3.9+
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.9 or higher"
        errors=$((errors + 1))
    else
        PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MIN_VERSION="3.9"
        
        if [ "$(printf '%s\n' "$MIN_VERSION" "$PY_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
            print_error "Python 3.9 or higher is required. You have version $PY_VERSION"
            errors=$((errors + 1))
        else
            print_status "Python $PY_VERSION found"
        fi
    fi
    
    # Check for pip
    if ! python3 -m pip --version &> /dev/null; then
        print_error "pip is not installed"
        errors=$((errors + 1))
    fi
    
    return $errors
}

# Get latest release version
get_latest_version() {
    # For now, we'll use the main branch
    # In the future, this could query git tags or releases
    echo "main"
}

# Clone Albert repository
download_albert() {
    local version="$1"
    
    print_info "Cloning Albert ${version}..."
    
    # Check if install directory exists
    if [ -d "$INSTALL_DIR" ]; then
        print_warning "Installation directory already exists: $INSTALL_DIR"
        read -p "Remove existing installation? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$INSTALL_DIR"
        else
            print_info "Installation cancelled"
            return 1
        fi
    fi
    
    # Clone repository
    if [ "$version" = "main" ]; then
        # Clone main branch
        if ! git clone "${REPO_URL}" "$INSTALL_DIR"; then
            print_error "Failed to clone Albert repository"
            return 1
        fi
    else
        # Clone and checkout specific version
        if ! git clone "${REPO_URL}" "$INSTALL_DIR"; then
            print_error "Failed to clone Albert repository"
            return 1
        fi
        
        cd "$INSTALL_DIR"
        if ! git checkout "$version" 2>/dev/null; then
            print_warning "Version $version not found, using main branch"
        fi
        cd - > /dev/null
    fi
    
    print_status "Albert cloned to $INSTALL_DIR"
    return 0
}

# Setup Python environment
setup_environment() {
    print_info "Setting up Python environment..."
    
    cd "$INSTALL_DIR"
    
    # Create virtual environment
    if ! python3 -m venv .venv; then
        print_error "Failed to create virtual environment"
        return 1
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip
    print_info "Upgrading pip..."
    python3 -m pip install --upgrade pip --quiet
    
    # Install dependencies
    print_info "Installing dependencies (this may take a few minutes)..."
    if [ -f requirements.txt ]; then
        python3 -m pip install -r requirements.txt --quiet
        print_status "Dependencies installed"
    else
        print_warning "requirements.txt not found"
    fi
    
    return 0
}

# Create launcher script
create_launcher() {
    print_info "Creating launcher script..."
    
    mkdir -p "$BIN_DIR"
    
    cat > "$BIN_DIR/albert" << 'EOF'
#!/bin/bash
# Albert launcher script
# <reason>chain: Launcher script that handles environment activation and command routing</reason>

ALBERT_HOME="${ALBERT_HOME:-$HOME/.albert}"

# Check if Albert is installed
if [ ! -d "$ALBERT_HOME" ]; then
    echo "Error: Albert not found at $ALBERT_HOME"
    echo "Please run the installer first:"
    echo "  curl -fsSL https://github.com/pimdewitte/albert/releases/download/stable/download_cli.sh | bash"
    exit 1
fi

# Activate virtual environment
if [ -f "$ALBERT_HOME/.venv/bin/activate" ]; then
    source "$ALBERT_HOME/.venv/bin/activate"
else
    echo "Error: Virtual environment not found"
    echo "Please reinstall Albert"
    exit 1
fi

# Change to Albert directory
cd "$ALBERT_HOME"

# Handle commands
case "$1" in
    update)
        echo "Updating Albert..."
        git pull origin main
        source "$ALBERT_HOME/.venv/bin/activate"
        python3 -m pip install -r requirements.txt --quiet
        echo "Albert updated successfully!"
        ;;
    
    version)
        echo "Albert v$(cat VERSION 2>/dev/null || echo 'unknown')"
        ;;
    
    help|--help|-h|"")
        # Pass through to Python for help
        python -m physics_agent.__main__ --help
        ;;
    
    validate)
        # Keep validate for backward compatibility
        shift
        python physics_agent/run_validator.py "$@"
        ;;
    
    *)
        # Pass all other commands to Python
        python -m physics_agent.__main__ "$@"
        ;;
esac
EOF
    
    chmod +x "$BIN_DIR/albert"
    
    # Check if bin directory is in PATH
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        print_warning "$BIN_DIR is not in your PATH"
        
        # Detect shell and add to appropriate config file
        local shell_config=""
        if [ -n "$BASH_VERSION" ]; then
            shell_config="$HOME/.bashrc"
        elif [ -n "$ZSH_VERSION" ]; then
            shell_config="$HOME/.zshrc"
        fi
        
        if [ -n "$shell_config" ]; then
            echo "" >> "$shell_config"
            echo "# Albert" >> "$shell_config"
            echo "export PATH=\"\$PATH:$BIN_DIR\"" >> "$shell_config"
            print_status "Added $BIN_DIR to PATH in $shell_config"
            print_info "Run 'source $shell_config' or restart your terminal"
        else
            print_info "Add this to your shell configuration:"
            echo "    export PATH=\"\$PATH:$BIN_DIR\""
        fi
    fi
    
    print_status "Launcher script created"
}

# Run configuration
run_configuration() {
    if [ "$CONFIGURE" = "true" ]; then
        print_info "Running initial configuration..."
        
        cd "$INSTALL_DIR"
        source .venv/bin/activate
        
        if python albert_setup.py; then
            print_status "Configuration complete"
        else
            print_warning "Configuration cancelled or failed"
            print_info "You can run 'albert setup' later to configure"
        fi
    else
        print_info "Skipping configuration (CONFIGURE=false)"
        print_info "Run 'albert setup' to configure Albert"
    fi
}

# Main installation flow
main() {
    print_banner
    
    # Check requirements
    if ! check_requirements; then
        print_error "System requirements not met"
        exit 1
    fi
    
    # Get version
    VERSION="${VERSION:-$(get_latest_version)}"
    print_info "Installing Albert ${VERSION}"
    
    # Download Albert
    if ! download_albert "$VERSION"; then
        print_error "Installation failed"
        exit 1
    fi
    
    # Setup environment
    if ! setup_environment; then
        print_error "Environment setup failed"
        exit 1
    fi
    
    # Create launcher
    create_launcher
    
    # Run configuration
    run_configuration
    
    # Success message
    echo ""
    print_status "${BOLD}Albert installed successfully!${NC}"
    echo ""
    echo "To get started:"
    echo "  1. Run 'albert help' to see available commands"
    echo "  2. Run 'albert setup' to configure API keys (if skipped)"
    echo "  3. Run 'albert discover' to start discovering new theories"
    echo ""
    echo "Happy discovering! 🚀"
}

# Run main installation
main "$@" 