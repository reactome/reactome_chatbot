# Using Vault

The files in this directory are to be used inside of a **`hashicorp/vault`**
    Docker container.

This project's `compose.yaml` mounts this directory to `/config`.


### README Overview

- [First-time setup](#first-time-setup)
    + Start here for new deployments.
- [Unsealing Vault (Nth-time setup)](#unsealing-vault-nth-time-setup)
    +
- [What is Vault?](#what-is-vault)


## First-time setup

These steps are only required for an initial deployment of Vault for Postgres
    credential-management usage.

1. Initialize the vault root token and unseal keys.
    - Copy the token and keys somewhere safe for now.
    - 🛑 _**Treat these as you would a sensitive password!
        Especially the root token!**_ 🛑
```sh
/config/1-init-vault.sh  # follow the prompts
```

2. Vault will always startup in a **sealed** state. Unseal it.
    - Repeat the command with different keys until the unseal threshold is reached.
    - This is required every time the container is restarted.
```sh
vault operator unseal  # paste an unseal key when prompted
```

3. Login using the root token (just this once).
```sh
export VAULT_TOKEN=$(vault login -token-only)  # paste the root token when prompted
```

4. Setup Vault policies, database secrets engine, and Postgres roles.
```sh
/config/4-setup-postgres.sh
```

5. Distribute unseal keys to team members. Do not store them all in one place.
    - Consider deleting (or at least encrypting) the root token.
    - ⚠️ _**Losing a quorum of unseal keys and the root token means a loss of all
        Vault contents, including Postgres access.**_ ⚠️


## Unsealing Vault (Nth-time setup)

Use the `vault operator unseal` command with unseal keys (repeatedly for the
    threshold number of unseal keys), as described in the
    [above](#first-time-setup) section.


## What is Vault?

[Hashicorp Vault](https://developer.hashicorp.com/vault) is a secrets manager
for protecting sensitive information such as
- authentication tokens,
- API keys, and
- database credentials.

Vault also features a
[database secrets engine](https://developer.hashicorp.com/vault/docs/secrets/databases)
which we leverage to issue time-limited credentials to our application.
