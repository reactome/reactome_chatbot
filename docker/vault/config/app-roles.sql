CREATE ROLE "{{name}}" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}';

-- Role can connect and create within chainlit and langgraph
GRANT
    CREATE,
    CONNECT
ON DATABASE chainlit, langgraph TO "{{name}}";

GRANT
    SELECT,
    INSERT,
    UPDATE,
    DELETE,
    REFERENCES
ON ALL TABLES IN SCHEMA public TO "{{name}}";
