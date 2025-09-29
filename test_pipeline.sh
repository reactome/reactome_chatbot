#!/bin/bash

# Test Pipeline for TP53-focused Reactome KG and Embeddings
# This script creates a complete test environment without touching existing data

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_COMPOSE_FILE="docker-compose.test.yml"
TEST_NEO4J_PORT="7688"
TEST_WEAVIATE_PORT="8081"
TEST_NEO4J_HTTP_PORT="7475"

# Timing
STARTUP_DELAY=30
HEALTH_CHECK_DELAY=10

echo -e "${BLUE}🧪 Starting TP53 Test Pipeline${NC}"
echo "=================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to check if ports are available
check_ports() {
    print_info "Checking if test ports are available..."
    
    if lsof -Pi :$TEST_NEO4J_PORT -sTCP:LISTEN -t >/dev/null; then
        print_error "Port $TEST_NEO4J_PORT is already in use"
        exit 1
    fi
    
    if lsof -Pi :$TEST_WEAVIATE_PORT -sTCP:LISTEN -t >/dev/null; then
        print_error "Port $TEST_WEAVIATE_PORT is already in use"
        exit 1
    fi
    
    if lsof -Pi :$TEST_NEO4J_HTTP_PORT -sTCP:LISTEN -t >/dev/null; then
        print_error "Port $TEST_NEO4J_HTTP_PORT is already in use"
        exit 1
    fi
    
    print_status "All test ports are available"
}

# Function to start test services
start_test_services() {
    print_info "Starting test services..."
    
    if [ ! -f "$TEST_COMPOSE_FILE" ]; then
        print_error "Test compose file not found: $TEST_COMPOSE_FILE"
        exit 1
    fi
    
    docker-compose -f "$TEST_COMPOSE_FILE" up -d
    
    print_status "Test services started"
    print_info "Waiting $STARTUP_DELAY seconds for services to initialize..."
    sleep $STARTUP_DELAY
}

# Function to check service health
check_service_health() {
    print_info "Checking service health..."
    
    # Check Neo4j
    print_info "Checking Neo4j test service..."
    for i in {1..10}; do
        if docker exec reactome-neo4j-test cypher-shell -u neo4j -p reactome-test "RETURN 1" >/dev/null 2>&1; then
            print_status "Neo4j test service is healthy"
            break
        fi
        if [ $i -eq 10 ]; then
            print_error "Neo4j test service failed to start properly"
            exit 1
        fi
        print_info "Waiting for Neo4j... (attempt $i/10)"
        sleep $HEALTH_CHECK_DELAY
    done
    
    # Check Weaviate
    print_info "Checking Weaviate test service..."
    for i in {1..10}; do
        if curl -f http://localhost:$TEST_WEAVIATE_PORT/v1/meta >/dev/null 2>&1; then
            print_status "Weaviate test service is healthy"
            break
        fi
        if [ $i -eq 10 ]; then
            print_error "Weaviate test service failed to start properly"
            exit 1
        fi
        print_info "Waiting for Weaviate... (attempt $i/10)"
        sleep $HEALTH_CHECK_DELAY
    done
}

# Function to build test knowledge graph
build_test_kg() {
    print_info "Building FULL Reactome knowledge graph (following create_reactome_kg.sh pattern)..."
    
    # Step 1: Start Reactome source database
    print_info "Starting Reactome source database..."
    docker run -d \
        --name "reactome-source-test" \
        -p "7476:7474" \
        -p "7689:7687" \
        -e NEO4J_AUTH=neo4j/reactome \
        -e NEO4J_dbms_memory_heap_maxSize=8g \
        public.ecr.aws/reactome/graphdb:latest
    
    print_info "Waiting for Reactome source to initialize..."
    
    # Wait for the container to be ready with proper health check
    for i in {1..30}; do
        # Try to connect using curl to the HTTP endpoint first
        if curl -f http://localhost:7476 >/dev/null 2>&1; then
            print_status "Reactome source HTTP endpoint is ready!"
            # Give it a bit more time for the Bolt protocol to be ready
            sleep 30
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Reactome source failed to start properly after 5 minutes"
            exit 1
        fi
        print_info "Waiting for Reactome source... (attempt $i/30)"
        sleep 10
    done
    
    # Step 2: Extract data using the existing build_knowledge_graph.py
    print_info "Extracting data from Reactome source..."
    cd src/data_generation/reactome_kg
    
    # Set environment variables to point to the test source
    export NEO4J_URI="bolt://localhost:7689"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="reactome"
    
    # Run the existing KG builder
    python3 build_knowledge_graph.py
    
    if [ $? -eq 0 ]; then
        print_status "Data extraction completed successfully"
    else
        print_error "Failed to extract data from Reactome source"
        exit 1
    fi
    
    # Step 3: Create test KG container and import
    print_info "Creating test knowledge graph container..."
    docker run -d \
        --name "reactome-kg-test" \
        -p "7690:7687" \
        -p "7478:7474" \
        -e NEO4J_AUTH=neo4j/reactome-test \
        -e NEO4J_dbms_default__database=reactome-kg-test \
        -e NEO4J_PLUGINS='["apoc","graph-data-science","n10s"]' \
        -e NEO4J_dbms_security_procedures_unrestricted="apoc.*,gds.*,n10s.*" \
        -e NEO4J_dbms_security_procedures_allowlist="apoc.*,gds.*,n10s.*" \
        -e NEO4J_apoc_import_file_enabled="true" \
        -e NEO4J_apoc_export_file_enabled="true" \
        -e NEO4J_apoc_import_file_use__neo4j__config="true" \
        -e NEO4J_dbms_memory_heap_initial__size=2G \
        -e NEO4J_dbms_memory_heap_max__size=4G \
        -e NEO4J_dbms_memory_pagecache_size=2G \
        neo4j:5.15-community
    
    print_info "Waiting 120 seconds for test KG container to start..."
    sleep 120
    
    # Copy CSV files to container
    print_info "Copying CSV files to test container..."
    docker cp biocypher-out/enhanced_reactome_kg/. reactome-kg-test:/data/
    
    # Import the knowledge graph
    print_info "Importing knowledge graph into test container..."
    docker stop reactome-kg-test
    
    # Use the generated import script with better memory settings
    print_info "Using generated Neo4j import script with enhanced settings..."
    docker run --rm \
        --name "reactome-kg-test-import" \
        --volumes-from reactome-kg-test \
        -v "$(pwd)/biocypher-out/enhanced_reactome_kg:/import" \
        -e NEO4J_dbms_memory_heap_initial__size=2G \
        -e NEO4J_dbms_memory_heap_max__size=4G \
        -e NEO4J_dbms_memory_pagecache_size=2G \
        neo4j:5.15-community \
        bash -c "
        # Set memory limits for import
        export NEO4J_dbms_memory_heap_initial__size=2G
        export NEO4J_dbms_memory_heap_max__size=4G
        export NEO4J_dbms_memory_pagecache_size=2G
        
        # Run the import with verbose output and better error handling
        bin/neo4j-admin database import full \
        --delimiter=';' \
        --array-delimiter='|' \
        --quote='@' \
        --overwrite-destination=true \
        --skip-bad-relationships=true \
        --bad-tolerance=10000 \
        --verbose \
        --nodes='/data/Pathway-header.csv,/data/Pathway-part.*' \
        --nodes='/data/Disease-header.csv,/data/Disease-part.*' \
        --nodes='/data/Reaction-header.csv,/data/Reaction-part.*' \
        --nodes='/data/GeneProduct-header.csv,/data/GeneProduct-part.*' \
        --nodes='/data/Drug-header.csv,/data/Drug-part.*' \
        --nodes='/data/SmallMolecule-header.csv,/data/SmallMolecule-part.*' \
        --nodes='/data/MolecularComplex-header.csv,/data/MolecularComplex-part.*' \
        --relationships='/data/ActsOn-header.csv,/data/ActsOn-part.*' \
        --relationships='/data/SubPathwayOf-header.csv,/data/SubPathwayOf-part.*' \
        --relationships='/data/AssociatedWith-header.csv,/data/AssociatedWith-part.*' \
        --relationships='/data/HasCatalyst-header.csv,/data/HasCatalyst-part.*' \
        --relationships='/data/HasInput-header.csv,/data/HasInput-part.*' \
        --relationships='/data/Precedes-header.csv,/data/Precedes-part.*' \
        --relationships='/data/Treats-header.csv,/data/Treats-part.*' \
        --relationships='/data/HasOutput-header.csv,/data/HasOutput-part.*' \
        --relationships='/data/HasComponent-header.csv,/data/HasComponent-part.*' \
        --relationships='/data/PartOf-header.csv,/data/PartOf-part.*' \
        --relationships='/data/HasVariant-header.csv,/data/HasVariant-part.*' \
        reactome-kg-test
        "
    
    if [ $? -eq 0 ]; then
        print_status "Knowledge graph import completed"
        docker start reactome-kg-test
        print_info "Waiting 30 seconds for container to restart..."
        sleep 30
    else
        print_error "Knowledge graph import failed"
        docker start reactome-kg-test
        exit 1
    fi
    
    # Clean up source container
    print_info "Cleaning up Reactome source container..."
    docker stop reactome-source-test
    docker rm reactome-source-test
    
    cd - >/dev/null
    print_status "Full Reactome knowledge graph built successfully in test environment"
}

# Function to create test embeddings
create_test_embeddings() {
    print_info "Creating TP53-focused embeddings from full KG (~100 nodes)..."
    
    cd src/data_generation/reactome_kg
    
    # Check if OpenAI API key is set
    if [ -z "$OPENAI_API_KEY" ]; then
        print_error "OPENAI_API_KEY environment variable is not set"
        print_info "Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'"
        exit 1
    fi
    
    # Run the test embedding creator (filters for TP53 from full KG)
    python3 create_test_embeddings.py
    
    if [ $? -eq 0 ]; then
        print_status "TP53-focused embeddings created successfully"
    else
        print_error "Failed to create test embeddings"
        exit 1
    fi
    
    cd - >/dev/null
}

# Function to verify test setup
verify_test_setup() {
    print_info "Verifying test setup..."
    
    # Check Neo4j node count
    print_info "Checking Neo4j test database..."
    NODE_COUNT=$(docker exec reactome-kg-test cypher-shell -u neo4j -p reactome-test -d reactome-kg-test "MATCH (n) RETURN count(n) as count" --format plain | tail -1)
    print_status "Neo4j test database contains $NODE_COUNT nodes"
    
    # Check Weaviate object count
    print_info "Checking Weaviate test database..."
    WEAVIATE_COUNT=$(curl -s "http://localhost:$TEST_WEAVIATE_PORT/v1/objects?class=TestReactomeKG&limit=1" | grep -o '"totalResults":[0-9]*' | cut -d':' -f2)
    if [ -n "$WEAVIATE_COUNT" ]; then
        print_status "Weaviate test database contains $WEAVIATE_COUNT objects"
    else
        print_warning "Could not retrieve Weaviate object count"
    fi
}

# Function to show access information
show_access_info() {
    print_info "Test environment access information:"
    echo ""
    echo "Neo4j Test Database:"
    echo "  - HTTP: http://localhost:7478"
    echo "  - Bolt: bolt://localhost:7690"
    echo "  - Username: neo4j"
    echo "  - Password: reactome-test"
    echo "  - Database: reactome-kg-test"
    echo "  - Container: reactome-kg-test"
    echo ""
    echo "Weaviate Test Database:"
    echo "  - REST API: http://localhost:$TEST_WEAVIATE_PORT"
    echo "  - Class: TestReactomeKG"
    echo ""
    echo "To stop test services:"
    echo "  docker-compose -f $TEST_COMPOSE_FILE down"
    echo ""
    echo "To remove test data volumes:"
    echo "  docker-compose -f $TEST_COMPOSE_FILE down -v"
}

# Function to cleanup on exit
cleanup() {
    print_info "Cleaning up test environment..."
    
    # Stop and remove test containers
    docker stop reactome-neo4j-test reactome-weaviate-test reactome-kg-test reactome-source-test 2>/dev/null || true
    docker rm reactome-neo4j-test reactome-weaviate-test reactome-kg-test reactome-source-test 2>/dev/null || true
    
    # Stop Docker Compose services
    docker-compose -f "$TEST_COMPOSE_FILE" down 2>/dev/null || true
    
    print_status "Test environment cleaned up"
}

# Main execution
main() {
    # Set trap for cleanup on script exit
    trap cleanup EXIT
    
    # Check prerequisites
    check_ports
    
    # Start Weaviate service only (Neo4j will be created by build_test_kg)
    print_info "Starting Weaviate test service..."
    docker-compose -f "$TEST_COMPOSE_FILE" up -d weaviate-test
    
    print_info "Waiting 30 seconds for Weaviate to initialize..."
    sleep 30
    
    # Check Weaviate health
    print_info "Checking Weaviate test service..."
    for i in {1..10}; do
        if curl -f http://localhost:$TEST_WEAVIATE_PORT/v1/meta >/dev/null 2>&1; then
            print_status "Weaviate test service is healthy"
            break
        fi
        if [ $i -eq 10 ]; then
            print_error "Weaviate test service failed to start properly"
            exit 1
        fi
        print_info "Waiting for Weaviate... (attempt $i/10)"
        sleep 10
    done
    
    # Build knowledge graph
    build_test_kg
    
    # Create embeddings
    create_test_embeddings
    
    # Verify setup
    verify_test_setup
    
    # Show access information
    show_access_info
    
    print_status "Full Reactome KG + TP53 embeddings test pipeline completed successfully!"
    print_info "You now have:"
    print_info "  - Full Reactome KG (72K+ nodes) in test Neo4j database"
    print_info "  - TP53-focused embeddings (~100 nodes) in test Weaviate database"
    print_info "Test services are running. Press Ctrl+C to stop and cleanup."
    
    # Keep services running until user stops
    while true; do
        sleep 60
    done
}

# Run main function
main "$@"
