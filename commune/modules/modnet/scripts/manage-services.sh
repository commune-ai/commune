#!/bin/bash

# Mod-Net Service Management Script
# Manages all PM2 services for Mod-Net infrastructure
# Author: Infrastructure Team
# Date: August 13, 2025

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMMUNE_IPFS_DIR="$PROJECT_ROOT/commune-ipfs"

# Function to print colored output (defined early for use in config loading)
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Check for required dependencies
check_dependencies() {
    if ! command -v jq >/dev/null 2>&1; then
        print_status $YELLOW "⚠️  jq is not installed. For more reliable service status checks, install jq:"
        print_status $BLUE "   Ubuntu/Debian: sudo apt-get install jq"
        print_status $BLUE "   CentOS/RHEL: sudo yum install jq"
        print_status $BLUE "   macOS: brew install jq"
        print_status $YELLOW "   Falling back to grep-based service status checks."
        echo
    fi
}

# Load environment variables if .env file exists
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a  # automatically export all variables
    source "$PROJECT_ROOT/.env"
    set +a  # stop automatically exporting
    print_status $GREEN "✅ Loaded configuration from .env file"
else
    print_status $YELLOW "⚠️  No .env file found. Using default values. Copy .env.example to .env and configure."
    # Set default values
    EXTERNAL_IP="[CONFIGURE_IN_.ENV]"
    NGROK_CHAIN_RPC_DOMAIN="[CONFIGURE_NGROK_DOMAINS]"
    NGROK_IPFS_WEBUI_DOMAIN="[CONFIGURE_NGROK_DOMAINS]"
    NGROK_IPFS_UPLOADER_DOMAIN="[CONFIGURE_NGROK_DOMAINS]"
    NGROK_BLOCKCHAIN_EXPLORER_DOMAIN="[CONFIGURE_NGROK_DOMAINS]"
fi

# Service definitions
declare -A SERVICES=(
    ["chain"]="cd $PROJECT_ROOT && ./target/release/solochain-template-node --dev --rpc-external --rpc-cors all"
    ["ipfs"]="ipfs daemon"
    ["ipfs-worker"]="cd $COMMUNE_IPFS_DIR && COMMUNE_IPFS_HOST=0.0.0.0 COMMUNE_IPFS_PORT=8003 uv run python main.py"
    ["blockchain-explorer"]="cd $PROJECT_ROOT/webui && python3 -m http.server 8081"
    ["ngrok"]="ngrok start --all --config /home/com/.config/ngrok/ngrok.yml"
)

# Function to print colored output (duplicate removed - defined earlier)

# Function to check if service is running
is_service_running() {
    local service_name=$1
    # Check if jq is available, fallback to grep if not
    if command -v jq >/dev/null 2>&1; then
        pm2 jlist | jq -e --arg name "$service_name" '.[] | select(.name == $name and .pm2_env.status == "online")' >/dev/null 2>&1
    else
        pm2 describe "$service_name" >/dev/null 2>&1 && \
        [[ $(pm2 describe "$service_name" | grep -c "status.*online") -gt 0 ]]
    fi
}

# Function to start a single service
start_service() {
    local service_name=$1
    local command=${SERVICES[$service_name]}

    if is_service_running "$service_name"; then
        print_status $YELLOW "⚠️  Service '$service_name' is already running"
        return 0
    fi

    print_status $BLUE "🚀 Starting service: $service_name"

    case $service_name in
        "chain")
            # Check if blockchain binary exists
            if [[ ! -f "$PROJECT_ROOT/target/release/solochain-template-node" ]]; then
                print_status $RED "❌ Blockchain binary not found. Please build the project first:"
                print_status $RED "   cd $PROJECT_ROOT && cargo build --release"
                return 1
            fi
            ;;
        "ipfs")
            # Check if IPFS is initialized
            if ! ipfs id >/dev/null 2>&1; then
                print_status $YELLOW "🔧 Initializing IPFS..."
                ipfs init
                # Configure IPFS for external access
                ipfs config Addresses.API /ip4/0.0.0.0/tcp/5001
                ipfs config Addresses.Gateway /ip4/0.0.0.0/tcp/8080
                ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["*"]'
                ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods '["GET", "POST"]'
            fi
            ;;
        "ipfs-worker")
            # Check if commune-ipfs directory exists and dependencies are installed
            if [[ ! -d "$COMMUNE_IPFS_DIR" ]]; then
                print_status $RED "❌ commune-ipfs directory not found at $COMMUNE_IPFS_DIR"
                return 1
            fi

            # Install dependencies if needed
            if [[ ! -d "$COMMUNE_IPFS_DIR/.venv" ]]; then
                print_status $YELLOW "🔧 Installing IPFS worker dependencies..."
                cd "$COMMUNE_IPFS_DIR"
                uv sync
            fi
            ;;
        "ngrok")
            # Check if ngrok config exists
            if [[ ! -f "/home/com/.config/ngrok/ngrok.yml" ]]; then
                print_status $RED "❌ Ngrok configuration not found at /home/com/.config/ngrok/ngrok.yml"
                return 1
            fi
            ;;
    esac

    # Start the service
    if pm2 start "$command" --name "$service_name" >/dev/null 2>&1; then
        print_status $GREEN "✅ Service '$service_name' started successfully"

        # Wait a moment for service to initialize
        sleep 2

        # Verify service is running
        if is_service_running "$service_name"; then
            print_status $GREEN "✅ Service '$service_name' is running and healthy"
        else
            print_status $YELLOW "⚠️  Service '$service_name' started but may have issues. Check logs: pm2 logs $service_name"
        fi
    else
        print_status $RED "❌ Failed to start service '$service_name'"
        return 1
    fi
}

# Function to stop a single service
stop_service() {
    local service_name=$1

    if ! is_service_running "$service_name"; then
        print_status $YELLOW "⚠️  Service '$service_name' is not running"
        return 0
    fi

    print_status $BLUE "🛑 Stopping service: $service_name"

    if pm2 stop "$service_name" >/dev/null 2>&1; then
        print_status $GREEN "✅ Service '$service_name' stopped successfully"
    else
        print_status $RED "❌ Failed to stop service '$service_name'"
        return 1
    fi
}

# Function to restart a single service
restart_service() {
    local service_name=$1

    print_status $BLUE "🔄 Restarting service: $service_name"

    if pm2 restart "$service_name" >/dev/null 2>&1; then
        print_status $GREEN "✅ Service '$service_name' restarted successfully"

        # Wait a moment for service to initialize
        sleep 2

        # Verify service is running
        if is_service_running "$service_name"; then
            print_status $GREEN "✅ Service '$service_name' is running and healthy"
        else
            print_status $YELLOW "⚠️  Service '$service_name' restarted but may have issues. Check logs: pm2 logs $service_name"
        fi
    else
        print_status $RED "❌ Failed to restart service '$service_name'"
        return 1
    fi
}

# Function to show service status
show_status() {
    print_status $BLUE "📊 Service Status:"
    echo
    pm2 list
    echo

    # Show individual service health
    for service in "${!SERVICES[@]}"; do
        if is_service_running "$service"; then
            print_status $GREEN "✅ $service: Running"
        else
            print_status $RED "❌ $service: Not running"
        fi
    done
}

# Function to show service logs
show_logs() {
    local service_name=${1:-"all"}

    if [[ "$service_name" == "all" ]]; then
        print_status $BLUE "📋 Showing logs for all services (press Ctrl+C to exit):"
        pm2 logs
    else
        if [[ -n "${SERVICES[$service_name]}" ]]; then
            print_status $BLUE "📋 Showing logs for service: $service_name"
            pm2 logs "$service_name"
        else
            print_status $RED "❌ Unknown service: $service_name"
            print_status $BLUE "Available services: ${!SERVICES[*]}"
            return 1
        fi
    fi
}

# Function to start all services
start_all() {
    print_status $BLUE "🚀 Starting all Mod-Net services..."
    echo

    # Start services in order (dependencies first)
    local services_order=("ipfs" "chain" "ipfs-worker" "ngrok")
    local failed_services=()

    for service in "${services_order[@]}"; do
        if ! start_service "$service"; then
            failed_services+=("$service")
        fi
        echo
    done

    echo
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        print_status $GREEN "🎉 All services started successfully!"
        print_status $BLUE "📊 Final status:"
        show_status
    else
        print_status $RED "❌ Some services failed to start: ${failed_services[*]}"
        print_status $BLUE "📊 Current status:"
        show_status
        return 1
    fi
}

# Function to stop all services
stop_all() {
    print_status $BLUE "🛑 Stopping all Mod-Net services..."
    echo

    # Stop services in reverse order
    local services_order=("ngrok" "ipfs-worker" "chain" "ipfs")

    for service in "${services_order[@]}"; do
        stop_service "$service"
        echo
    done

    print_status $GREEN "✅ All services stopped"
}

# Function to restart all services
restart_all() {
    print_status $BLUE "🔄 Restarting all Mod-Net services..."
    echo

    stop_all
    echo
    start_all
}

# Function to test connectivity
test_connectivity() {
    print_status $BLUE "🔍 Testing service connectivity..."
    echo

    # Test blockchain RPC
    print_status $BLUE "Testing Blockchain RPC..."
    if curl -s -H "Content-Type: application/json" \
       -d '{"id":1, "jsonrpc":"2.0", "method": "system_health", "params":[]}' \
       http://localhost:9944 >/dev/null 2>&1; then
        print_status $GREEN "✅ Blockchain RPC: Responsive"
    else
        print_status $RED "❌ Blockchain RPC: Not responding"
    fi

    # Test IPFS API
    print_status $BLUE "Testing IPFS API..."
    if curl -s -X POST http://localhost:5001/api/v0/version >/dev/null 2>&1; then
        print_status $GREEN "✅ IPFS API: Responsive"
    else
        print_status $RED "❌ IPFS API: Not responding"
    fi

    # Test IPFS Worker
    print_status $BLUE "Testing IPFS Worker..."
    if curl -s http://localhost:8003/health >/dev/null 2>&1; then
        print_status $GREEN "✅ IPFS Worker: Responsive"
    else
        print_status $RED "❌ IPFS Worker: Not responding"
    fi

    # Test IPFS Upload Frontend
    print_status $BLUE "Testing IPFS Upload Frontend..."
    if curl -s http://localhost:8003/ >/dev/null 2>&1; then
        print_status $GREEN "✅ IPFS Upload Frontend: Responsive"
    else
        print_status $RED "❌ IPFS Upload Frontend: Not responding"
    fi

    # Test Blockchain Explorer
    print_status $BLUE "Testing Blockchain Explorer..."
    if curl -s http://localhost:8081/blockchain-explorer.html >/dev/null 2>&1; then
        print_status $GREEN "✅ Blockchain Explorer: Responsive"
    else
        print_status $RED "❌ Blockchain Explorer: Not responding"
    fi

    echo
    print_status $BLUE "🌐 External connectivity test:"
    print_status $BLUE "Direct access: http://${EXTERNAL_IP}:${EXTERNAL_CHAIN_RPC_PORT} (Blockchain RPC)"
    print_status $BLUE "Direct access: http://${EXTERNAL_IP}:${EXTERNAL_IPFS_API_PORT} (IPFS API)"
    print_status $BLUE "Direct access: http://${EXTERNAL_IP}:${EXTERNAL_IPFS_WORKER_PORT} (IPFS Worker + Upload Frontend)"
    print_status $BLUE "Direct access: http://${EXTERNAL_IP}:${EXTERNAL_BLOCKCHAIN_EXPLORER_PORT} (Blockchain Explorer)"
    echo
    print_status $BLUE "🎨 WebUI Access:"
    print_status $BLUE "IPFS WebUI: ${NGROK_IPFS_WEBUI_DOMAIN}/webui/"
    print_status $BLUE "IPFS Upload Frontend: ${NGROK_IPFS_UPLOADER_DOMAIN}/"
    print_status $BLUE "Blockchain Explorer: ${NGROK_BLOCKCHAIN_EXPLORER_DOMAIN}/blockchain-explorer.html"
}

# Function to save PM2 configuration
save_config() {
    print_status $BLUE "💾 Saving PM2 configuration..."

    pm2 save

    # Setup PM2 startup script using robust detection
    pm2 startup --detect
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ PM2 startup configuration saved and auto-start enabled"
    else
        print_status $YELLOW "⚠️  PM2 could not automatically configure auto-start. Please run the command suggested by 'pm2 startup' manually."
        pm2 startup
    fi
}

# Function to show help
show_help() {
    echo "Mod-Net Service Management Script"
    echo
    echo "Usage: $0 [COMMAND] [SERVICE]"
    echo
    echo "Commands:"
    echo "  start [service]     Start a specific service or all services"
    echo "  stop [service]      Stop a specific service or all services"
    echo "  restart [service]   Restart a specific service or all services"
    echo "  status              Show status of all services"
    echo "  logs [service]      Show logs for a specific service or all services"
    echo "  test                Test connectivity to all services"
    echo "  save                Save PM2 configuration and setup auto-start"
    echo "  help                Show this help message"
    echo
    echo "Available services: ${!SERVICES[*]}"
    echo
    echo "Examples:"
    echo "  $0 start            # Start all services"
    echo "  $0 start chain      # Start only the blockchain service"
    echo "  $0 restart ipfs     # Restart only the IPFS service"
    echo "  $0 logs ipfs-worker # Show logs for IPFS worker"
    echo "  $0 status           # Show status of all services"
    echo "  $0 test             # Test connectivity"
}

# Main script logic
main() {
    local command=${1:-"help"}
    local service_name=$2

    # Check dependencies on first run
    check_dependencies

    case $command in
        "start")
            if [[ -n "$service_name" ]]; then
                if [[ -n "${SERVICES[$service_name]}" ]]; then
                    start_service "$service_name"
                else
                    print_status $RED "❌ Unknown service: $service_name"
                    print_status $BLUE "Available services: ${!SERVICES[*]}"
                    exit 1
                fi
            else
                start_all
            fi
            ;;
        "stop")
            if [[ -n "$service_name" ]]; then
                if [[ -n "${SERVICES[$service_name]}" ]]; then
                    stop_service "$service_name"
                else
                    print_status $RED "❌ Unknown service: $service_name"
                    print_status $BLUE "Available services: ${!SERVICES[*]}"
                    exit 1
                fi
            else
                stop_all
            fi
            ;;
        "restart")
            if [[ -n "$service_name" ]]; then
                if [[ -n "${SERVICES[$service_name]}" ]]; then
                    restart_service "$service_name"
                else
                    print_status $RED "❌ Unknown service: $service_name"
                    print_status $BLUE "Available services: ${!SERVICES[*]}"
                    exit 1
                fi
            else
                restart_all
            fi
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "$service_name"
            ;;
        "test")
            test_connectivity
            ;;
        "save")
            save_config
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_status $RED "❌ Unknown command: $command"
            echo
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
