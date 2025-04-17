path "sys/mounts/database" {
  capabilities = [ "create", "update" ]
}

path "database/*" {
  capabilities = [ "create", "update" ]
}
