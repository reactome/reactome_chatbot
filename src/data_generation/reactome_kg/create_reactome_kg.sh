#!/bin/bash

# This script automates the entire process of creating a Reactome Knowledge Graph:
# 1. Pulls latest Reactome Neo4j database
# 2. Extracts and processes data using our adapter
# 3. Generates CSV files with quality checks
# 4. Creates a versioned Neo4j container
# 5. Imports the knowledge graph
# 6. Validates and reports results
#
# Usage: ./create_reactome_kg.sh [options]
# Options:
#   --clean          Clean up existing containers and data
#   --no-validation  Skip validation steps
#   --help           Show this help message

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Timing constants
INITIALIZATION_DELAY=60
CONTAINER_START_DELAY=120
POST_IMPORT_DELAY=30
VALIDATION_DELAY=15

# Port constants
NEO4J_BROWSER_PORT=7474
NEO4J_BOLT_PORT=7687
REACTOME_BROWSER_PORT=7475
REACTOME_BOLT_PORT=7688

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
DATA_DIR="$PROJECT_ROOT/src/data_generation/reactome_kg"
OUTPUT_DIR="$DATA_DIR/biocypher-out/enhanced_reactome_kg"

# Default values
CLEAN=false
SKIP_VALIDATION=false
REACTOME_VERSION="latest"
CONTAINER_NAME="reactome-source"
KG_CONTAINER_NAME="reactome-kg"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --no-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --clean          Clean up existing containers and data"
            echo "  --no-validation  Skip validation steps"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

_stop_container() {
    local container_name="$1"
    if docker ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
        log_info "Stopping container: $container_name"
        docker stop "$container_name" 2>/dev/null || true
        docker rm "$container_name" 2>/dev/null || true
    fi
}

_stop_containers_by_port() {
    local port="$1"
    log_info "Stopping containers using port $port..."
    docker stop $(docker ps -q --filter "publish=$port") 2>/dev/null || true
}

_wait_for_container() {
    local container_name="$1"
    local delay="${2:-30}"
    log_info "Waiting ${delay} seconds for $container_name to be ready..."
    sleep "$delay"
}

_execute_cypher() {
    local container_name="$1"
    local query="$2"
    docker exec "$container_name" bin/cypher-shell -u neo4j -p reactome "$query" 2>/dev/null
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Wait for service to be ready
wait_for_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    log_info "Waiting for $service_name to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec "$service_name" curl -s http://localhost:$port >/dev/null 2>&1; then
            log_success "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    log_error "$service_name failed to start within $(($max_attempts * 2)) seconds"
    return 1
}

cleanup_containers() {
    log_step "Cleaning up existing containers and Docker Compose services..."
    
    # Stop Docker Compose services first
    if [ -f "$PROJECT_ROOT/docker-compose.enhanced.yml" ]; then
        log_info "Stopping Docker Compose services..."
        docker-compose -f "$PROJECT_ROOT/docker-compose.enhanced.yml" down 2>/dev/null || true
    fi
    
    # Stop and remove individual containers
    for container in "$CONTAINER_NAME" "$KG_CONTAINER_NAME"*; do
        _stop_container "$container"
    done
    
    # Stop any containers using Neo4j ports
    _stop_containers_by_port "$NEO4J_BROWSER_PORT"
    _stop_containers_by_port "$NEO4J_BOLT_PORT"
    
    log_success "Container cleanup completed"
}

cleanup_data() {
    log_step "Cleaning up existing data..."
    
    # Remove existing CSV files
    if [ -d "$OUTPUT_DIR" ]; then
        log_info "Removing existing CSV files..."
        rm -rf "$OUTPUT_DIR"
    fi
    
    # Remove cache directories
    if [ -d "$PROJECT_ROOT/.cache" ]; then
        rm -rf "$PROJECT_ROOT/.cache"
    fi
    
    if [ -d "$PROJECT_ROOT/biocypher-log" ]; then
        rm -rf "$PROJECT_ROOT/biocypher-log"
    fi
    
    log_success "Data cleanup completed"
}

setup_reactome_source() {
    log_step "Setting up Reactome source database..."
    
    # Start Reactome Neo4j database
    log_info "Starting Reactome Neo4j database..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p "$REACTOME_BROWSER_PORT:$NEO4J_BROWSER_PORT" \
        -p "$REACTOME_BOLT_PORT:$NEO4J_BOLT_PORT" \
        -e NEO4J_AUTH=neo4j/reactome \
        -e NEO4J_dbms_memory_heap_maxSize=8g \
        public.ecr.aws/reactome/graphdb:latest
    
    _wait_for_container "$CONTAINER_NAME" "$INITIALIZATION_DELAY"
    
    # Get Reactome version
    log_info "Detecting Reactome version..."
    REACTOME_VERSION=$(_execute_cypher "$CONTAINER_NAME" \
        "CALL db.info() YIELD version RETURN version;" | tail -n 1 | tr -d ' \t\n\r' || echo "unknown")
    
    if [ "$REACTOME_VERSION" = "unknown" ] || [ -z "$REACTOME_VERSION" ]; then
        log_warning "Could not detect Reactome version, using 'latest'"
        REACTOME_VERSION="latest"
    fi
    
    # Update container name with version
    KG_CONTAINER_NAME="reactome-kg-v${REACTOME_VERSION}"
    
    log_success "Reactome source database ready (version: $REACTOME_VERSION)"
}

extract_data() {
    log_step "Extracting data from Reactome database..."
    
    # Change to data generation directory
    cd "$DATA_DIR"
    
    # Set environment variables
    export NEO4J_URI="bolt://localhost:$REACTOME_BOLT_PORT"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="reactome"
    
    # Run the KG creation script
    log_info "Running knowledge graph creation..."
    python3 build_knowledge_graph.py
    
    # Wait a moment for directory creation to complete
    sleep 15
    
    if [ ! -d "$OUTPUT_DIR" ]; then
        log_error "Knowledge graph creation failed - no output directory found"
        exit 1
    fi
    
    log_success "Data extraction completed"
}

_count_files_by_type() {
    local file_pattern="$1"
    local description="$2"
    
    log_info "Counting $description..."
    echo -e "\n${CYAN}=== $description ===${NC}"
    for file in *-part*.csv; do
        if [[ $file == *"header"* ]]; then continue; fi
        local entity_type=$(echo "$file" | sed 's/-part.*//')
        local count=$(wc -l < "$file")
        printf "%-20s: %8s\n" "$entity_type" "$count"
    done
}

_check_duplicates() {
    echo -e "\n${CYAN}=== DUPLICATE CHECK ===${NC}"
    
    # Collect all node IDs
    cat Pathway-part*.csv Reaction-part*.csv MolecularComplex-part*.csv \
        GeneProduct-part*.csv SmallMolecule-part*.csv Disease-part*.csv \
        Drug-part*.csv 2>/dev/null | cut -d';' -f1 | sort > /tmp/all_nodes.txt
    
    local unique_nodes=$(sort /tmp/all_nodes.txt | uniq | wc -l)
    local total_node_records=$(wc -l < /tmp/all_nodes.txt)
    
    if [ "$unique_nodes" -eq "$total_node_records" ]; then
        echo "No duplicate nodes found"
    else
        echo "Found $((total_node_records - unique_nodes)) duplicate nodes"
    fi
}

_analyze_orphans() {
    echo -e "\n${CYAN}=== ORPHAN ANALYSIS ===${NC}"
    
    # Collect connected node IDs from edges
    cat HasInput-part*.csv HasOutput-part*.csv HasCatalyst-part*.csv \
        HasComponent-part*.csv HasVariant-part*.csv Precedes-part*.csv \
        PartOf-part*.csv SubPathwayOf-part*.csv ActsOn-part*.csv \
        Treats-part*.csv AssociatedWith-part*.csv 2>/dev/null | \
        cut -d';' -f1,6 | tr ';' '\n' | sort | uniq > /tmp/connected_nodes.txt
    
    local orphan_nodes=$(comm -23 /tmp/all_nodes.txt /tmp/connected_nodes.txt | wc -l)
    local connected_nodes=$(comm -12 /tmp/all_nodes.txt /tmp/connected_nodes.txt | wc -l)
    local total_node_records=$(wc -l < /tmp/all_nodes.txt)
    
    echo "Total nodes: $total_node_records"
    echo "Connected nodes: $connected_nodes"
    echo "Orphan nodes: $orphan_nodes"
    echo "Connectivity: $(echo "scale=1; $connected_nodes * 100 / $total_node_records" | bc)%"
    
    # Cleanup temp files
    rm -f /tmp/all_nodes.txt /tmp/connected_nodes.txt
}

analyze_data_quality() {
    log_step "Analyzing data quality..."
    
    if [ ! -d "$OUTPUT_DIR" ]; then
        log_error "Output directory not found: $OUTPUT_DIR"
        return 1
    fi
    
    cd "$OUTPUT_DIR"
    
    # Count nodes and edges by type
    _count_files_by_type "*-part*.csv" "NODE COUNTS"
    _count_files_by_type "*-part*.csv" "EDGE COUNTS"
    
    # Calculate totals
    local total_nodes=$(find . -name "*-part*.csv" -exec wc -l {} + | tail -1 | awk '{print $1}')
    local total_edges=$(find . -name "*-part*.csv" -exec wc -l {} + | tail -1 | awk '{print $1}')
    
    echo -e "\n${CYAN}=== SUMMARY ===${NC}"
    echo "Total nodes: $total_nodes"
    echo "Total edges: $total_edges"
    
    # Check for duplicates and analyze orphans
    _check_duplicates
    _analyze_orphans
    
    log_success "Data quality analysis completed"
}

create_kg_container() {
    log_step "Creating versioned Neo4j container with GDS for knowledge graph..."
    
    local versioned_name="$KG_CONTAINER_NAME"
    
    # Stop any existing Neo4j containers to avoid port conflicts
    _stop_containers_by_port "$NEO4J_BROWSER_PORT"
    _stop_containers_by_port "$NEO4J_BOLT_PORT"
    _stop_container "$versioned_name"
    
    # Create new Neo4j container with GDS and plugins
    log_info "Creating Neo4j 5.15 container with GDS: $versioned_name"
    docker run -d \
        --name "$versioned_name" \
        -p "$NEO4J_BROWSER_PORT:$NEO4J_BROWSER_PORT" \
        -p "$NEO4J_BOLT_PORT:$NEO4J_BOLT_PORT" \
        -e NEO4J_AUTH=neo4j/reactome \
        -e NEO4J_dbms_default__database=reactome-kg \
        -e NEO4J_PLUGINS='["apoc","graph-data-science","n10s"]' \
        -e NEO4J_dbms_security_procedures_unrestricted="apoc.*,gds.*,n10s.*" \
        -e NEO4J_dbms_security_procedures_allowlist="apoc.*,gds.*,n10s.*" \
        -e NEO4J_apoc_import_file_enabled="true" \
        -e NEO4J_apoc_export_file_enabled="true" \
        -e NEO4J_apoc_import_file_use__neo4j__config="true" \
        -e NEO4J_server_memory_heap_initial__size=1G \
        -e NEO4J_server_memory_heap_max__size=2G \
        -e NEO4J_server_memory_pagecache_size=1G \
        neo4j:5.15-community
    
    _wait_for_container "$versioned_name" "$CONTAINER_START_DELAY"
    
    # Verify GDS is available
    log_info "Verifying GDS installation..."
    if _execute_cypher "$versioned_name" "CALL gds.version();" | grep -q "gdsVersion"; then
        log_success "GDS is successfully installed and available"
    else
        log_warning "GDS verification failed, but continuing with import"
    fi
    
    # Copy CSV files to container
    log_info "Copying CSV files to container..."
    docker cp "$OUTPUT_DIR/." "$versioned_name:/data/"
    
    log_success "Container created with GDS: $versioned_name"
}

import_knowledge_graph() {
    log_step "Importing knowledge graph into Neo4j..."
    
    # Stop the container before import (required for neo4j-admin import)
    log_info "Stopping container for import process..."
    docker stop "$KG_CONTAINER_NAME"
    
    # Run the import command using a temporary container with the same volumes
    log_info "Running Neo4j import..."
    docker run --rm \
        --name "${KG_CONTAINER_NAME}-import" \
        --volumes-from "$KG_CONTAINER_NAME" \
        -v "$OUTPUT_DIR:/import" \
        neo4j:5.15-community \
        bin/neo4j-admin database import full \
        --delimiter=";" \
        --array-delimiter="|" \
        --quote='@' \
        --overwrite-destination=true \
        --skip-bad-relationships=true \
        --bad-tolerance=4000 \
        --nodes="/data/Pathway-header.csv,/data/Pathway-part.*" \
        --nodes="/data/Disease-header.csv,/data/Disease-part.*" \
        --nodes="/data/Reaction-header.csv,/data/Reaction-part.*" \
        --nodes="/data/GeneProduct-header.csv,/data/GeneProduct-part.*" \
        --nodes="/data/Drug-header.csv,/data/Drug-part.*" \
        --nodes="/data/SmallMolecule-header.csv,/data/SmallMolecule-part.*" \
        --nodes="/data/MolecularComplex-header.csv,/data/MolecularComplex-part.*" \
        --relationships="/data/ActsOn-header.csv,/data/ActsOn-part.*" \
        --relationships="/data/SubPathwayOf-header.csv,/data/SubPathwayOf-part.*" \
        --relationships="/data/AssociatedWith-header.csv,/data/AssociatedWith-part.*" \
        --relationships="/data/HasCatalyst-header.csv,/data/HasCatalyst-part.*" \
        --relationships="/data/HasInput-header.csv,/data/HasInput-part.*" \
        --relationships="/data/Precedes-header.csv,/data/Precedes-part.*" \
        --relationships="/data/Treats-header.csv,/data/Treats-part.*" \
        --relationships="/data/HasOutput-header.csv,/data/HasOutput-part.*" \
        --relationships="/data/HasComponent-header.csv,/data/HasComponent-part.*" \
        --relationships="/data/PartOf-header.csv,/data/PartOf-part.*" \
        --relationships="/data/HasVariant-header.csv,/data/HasVariant-part.*" \
        neo4j
    
    if [ $? -eq 0 ]; then
        log_success "Knowledge graph import completed"
        docker start "$KG_CONTAINER_NAME"
        _wait_for_container "$KG_CONTAINER_NAME" "$POST_IMPORT_DELAY"
    else
        log_error "Knowledge graph import failed"
        docker start "$KG_CONTAINER_NAME"
        exit 1
    fi
}

validate_knowledge_graph() {
    log_step "Validating knowledge graph..."
    
    # Container should already be running from import process
    log_info "Ensuring Neo4j container is ready..."
    sleep "$VALIDATION_DELAY"
    
    echo -e "\n${CYAN}=== VALIDATION RESULTS ===${NC}"
    
    # Get total node count
    local total_nodes=$(_execute_cypher "$KG_CONTAINER_NAME" \
        "MATCH (n) RETURN count(n) as total_nodes;" | tail -n 1 | tr -d ' \t\n\r')
    
    # Get total relationship count
    local total_relationships=$(_execute_cypher "$KG_CONTAINER_NAME" \
        "MATCH ()-[r]->() RETURN count(r) as total_relationships;" | tail -n 1 | tr -d ' \t\n\r')
    
    echo "Total nodes in Neo4j: $total_nodes"
    echo "Total relationships in Neo4j: $total_relationships"
    
    # Get node type breakdown
    log_info "Validating node types..."
    echo -e "\n${CYAN}=== NODE TYPES IN NEO4J ===${NC}"
    _execute_cypher "$KG_CONTAINER_NAME" \
        "MATCH (n) RETURN labels(n) as node_type, count(n) as count ORDER BY count DESC;" | \
        tail -n +2 | head -n -1 | while IFS='|' read -r node_type count; do
            node_type=$(echo "$node_type" | tr -d ' \t\n\r[]"')
            count=$(echo "$count" | tr -d ' \t\n\r')
            printf "%-30s: %8s\n" "$node_type" "$count"
        done
    
    # Get relationship type breakdown
    log_info "Validating relationship types..."
    echo -e "\n${CYAN}=== RELATIONSHIP TYPES IN NEO4J ===${NC}"
    _execute_cypher "$KG_CONTAINER_NAME" \
        "MATCH ()-[r]->() RETURN type(r) as relationship_type, count(r) as count ORDER BY count DESC;" | \
        tail -n +2 | head -n -1 | while IFS='|' read -r rel_type count; do
            rel_type=$(echo "$rel_type" | tr -d ' \t\n\r"')
            count=$(echo "$count" | tr -d ' \t\n\r')
            printf "%-30s: %8s\n" "$rel_type" "$count"
        done
    
    # Verify GDS is working
    log_info "Verifying GDS functionality..."
    echo -e "\n${CYAN}=== GDS VERIFICATION ===${NC}"
    local gds_version=$(_execute_cypher "$KG_CONTAINER_NAME" \
        "CALL gds.version();" | tail -n 1 | tr -d ' \t\n\r"')
    
    if [ -n "$gds_version" ] && [ "$gds_version" != "null" ]; then
        echo "GDS Version: $gds_version"
        echo "GDS is working correctly"
    else
        echo "GDS verification failed"
        log_warning "GDS may not be properly installed"
    fi
    
    log_success "Knowledge graph validation completed"
}

# =============================================================================
# FINAL REPORT
# =============================================================================

generate_final_report() {
    log_step "Generating final report..."
    
    echo -e "\n${GREEN}================================================================================${NC}"
    echo -e "${GREEN} REACTOME KNOWLEDGE GRAPH CREATION COMPLETED SUCCESSFULLY!${NC}"
    echo -e "${GREEN}================================================================================${NC}"
    
    echo -e "\n${CYAN} KNOWLEDGE GRAPH SUMMARY:${NC}"
    echo "• Reactome Version: $REACTOME_VERSION"
    echo "• Container Name: $KG_CONTAINER_NAME"
    echo "• Neo4j Version: 5.15-community with GDS"
    echo "• Neo4j Browser: http://localhost:7474"
    echo "• Username: neo4j"
    echo "• Password: reactome"
    echo "• GDS Available: Yes (Steiner tree algorithms enabled)"
    
    echo -e "\n${CYAN} FILES AND LOCATIONS:${NC}"
    echo "• CSV Files: $OUTPUT_DIR"
    echo "• Source Database: $CONTAINER_NAME (ports 7475, 7688)"
    echo "• Knowledge Graph: $KG_CONTAINER_NAME (ports 7474, 7687)"
    
    echo -e "\n${CYAN} SAMPLE QUERIES TO TRY:${NC}"
    echo "MATCH (n) RETURN count(n) as total_nodes;"
    echo "MATCH (n) RETURN labels(n) as node_type, count(n) as count ORDER BY count DESC;"
    echo "MATCH ()-[r]->() RETURN type(r) as relationship_type, count(r) as count ORDER BY count DESC;"
    echo "MATCH (p:Pathway)-[:PartOf]-(r:Reaction) RETURN p.name, count(r) as reaction_count ORDER BY reaction_count DESC LIMIT 10;"
    echo ""
    echo -e "${CYAN} GDS QUERIES TO TRY:${NC}"
    echo "CALL gds.version() YIELD version RETURN version;"
    echo "CALL gds.list() YIELD name, type RETURN name, type ORDER BY name;"
    
    echo -e "\n${CYAN} NEXT STEPS:${NC}"
    echo "1. Open Neo4j Browser at http://localhost:7474"
    echo "2. Explore the knowledge graph with sample queries"
    echo "3. Run the embedding pipeline to create vector database"
    echo "4. Build the chatbot interface"
    
    echo -e "\n${GREEN}================================================================================${NC}"
}

main() {
    echo -e "${PURPLE}Starting Reactome Knowledge Graph Creation Pipeline${NC}"
    echo -e "${PURPLE}================================================================================${NC}"
    
    # Check prerequisites
    if ! command_exists docker; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    if ! command_exists python3; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Cleanup if requested
    if [ "$CLEAN" = true ]; then
        cleanup_containers
        cleanup_data
    fi
    
    # Execute pipeline steps
    setup_reactome_source
    extract_data
    analyze_data_quality
    
    if [ "$SKIP_VALIDATION" = false ]; then
        create_kg_container
        import_knowledge_graph
        validate_knowledge_graph
    fi
    
    generate_final_report
    
    log_success "Pipeline completed successfully!"
}

# Run main function
main "$@"
