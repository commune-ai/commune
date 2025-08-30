#!/bin/bash

# Mod-Net External Port Configuration
# Maps datacenter external ports (2101-2201) to internal service ports

set -e

# Port mappings
# Note: Port 2001 is reserved for SSH forwarding (2001 -> 22)
declare -A PORT_MAP=(
    ["2101"]="30333"  # Blockchain P2P port
    ["2102"]="9944"   # Blockchain RPC port
    ["2103"]="9944"   # Blockchain WebSocket (same as RPC)
    ["2104"]="9615"   # Prometheus metrics
    ["2105"]="5001"   # IPFS API port
    ["2106"]="8080"   # IPFS Gateway port
    ["2107"]="4001"   # IPFS Swarm port
    ["2108"]="8003"   # IPFS Worker (off-chain bridge)
)

# Function to setup iptables port forwarding
setup_port_forwarding() {
    echo "Setting up port forwarding for Mod-Net services..."

    for external_port in "${!PORT_MAP[@]}"; do
        internal_port="${PORT_MAP[$external_port]}"

        echo "Forwarding external port $external_port to internal port $internal_port"

        # Forward incoming traffic from external port to internal port using DNAT
        EXTERNAL_IP=$(hostname -I | awk '{print $1}')
        sudo iptables -t nat -A PREROUTING -p tcp --dport $external_port -j DNAT --to-destination ${EXTERNAL_IP}:$internal_port

        # Allow traffic on external port
        sudo ufw allow $external_port/tcp

        echo "✓ Port $external_port -> $internal_port configured"
    done

    # Save iptables rules
    sudo iptables-save > /tmp/iptables-mod-net.rules
    echo "✓ iptables rules saved to /tmp/iptables-mod-net.rules"
}

# Function to remove port forwarding
remove_port_forwarding() {
    echo "Removing port forwarding for Mod-Net services..."

    for external_port in "${!PORT_MAP[@]}"; do
        internal_port="${PORT_MAP[$external_port]}"

        echo "Removing forwarding for port $external_port"

        # Remove iptables rule (ignore errors if rule doesn't exist)
        EXTERNAL_IP=$(hostname -I | awk '{print $1}')
        sudo iptables -t nat -D PREROUTING -p tcp --dport $external_port -j DNAT --to-destination ${EXTERNAL_IP}:$internal_port 2>/dev/null || true

        # Remove ufw rule
        sudo ufw delete allow $external_port/tcp 2>/dev/null || true

        echo "✓ Port $external_port forwarding removed"
    done
}

# Function to show current port status
show_port_status() {
    echo "Current Mod-Net port configuration:"
    echo "=================================="

    for external_port in "${!PORT_MAP[@]}"; do
        internal_port="${PORT_MAP[$external_port]}"

        # Check if port is listening
        if ss -ln | grep -q ":$internal_port "; then
            status="✓ LISTENING"
        else
            status="✗ NOT LISTENING"
        fi

        echo "External $external_port -> Internal $internal_port: $status"
    done

    echo ""
    echo "UFW Status:"
    sudo ufw status | grep -E "(2101|2102|2103|2104)" || echo "No external ports configured in UFW"

    echo ""
    echo "iptables NAT rules:"
    sudo iptables -t nat -L PREROUTING | grep -E "(2101|2102|2103|2104)" || echo "No NAT rules configured"
}

# Function to test external connectivity
test_connectivity() {
    echo "Testing external port connectivity..."

    for external_port in "${!PORT_MAP[@]}"; do
        internal_port="${PORT_MAP[$external_port]}"

        echo -n "Testing port $external_port: "

        # Test if we can connect to the external port
        if timeout 5 bash -c "</dev/tcp/localhost/$external_port" 2>/dev/null; then
            echo "✓ ACCESSIBLE"
        else
            echo "✗ NOT ACCESSIBLE"
        fi
    done
}

# Usage function
usage() {
    cat <<EOF
Usage: $0 [COMMAND]

Commands:
    setup     Setup port forwarding rules
    remove    Remove port forwarding rules
    status    Show current port status
    test      Test port connectivity
    help      Show this help message

Port Mappings:
    2001 -> 22    (SSH) - RESERVED
    2101 -> 30333 (Blockchain P2P)
    2102 -> 9944  (Blockchain RPC)
    2103 -> 9944  (Blockchain WebSocket)
    2104 -> 9615  (Prometheus)
    2105 -> 5001  (IPFS API)
    2106 -> 8080  (IPFS Gateway)
    2107 -> 4001  (IPFS Swarm)
    2108 -> 8003  (IPFS Worker)

Examples:
    $0 setup          # Setup all port forwarding
    $0 status         # Check current status
    $0 test           # Test connectivity
    $0 remove         # Remove all forwarding rules
EOF
}

# Main script logic
case "${1:-help}" in
    setup)
        setup_port_forwarding
        ;;
    remove)
        remove_port_forwarding
        ;;
    status)
        show_port_status
        ;;
    test)
        test_connectivity
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac
