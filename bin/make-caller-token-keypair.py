#!/usr/bin/env python3
"""Generate the keypair the answer endpoint verifies proof-of-human tokens with.

The endpoint holds only the **public** half, and verifies EdDSA or RS256 -- never
an HMAC algorithm, so a stolen public key cannot be turned into a signing key.
Whoever mints tokens holds the private half; who that is is D1 in
specs/010-search-page-answers, still open.

Rotation is this script plus a restart: generate, replace the public key the
service reads, hand the private half to the minter. Tokens signed by the old key
stop verifying immediately, which is the point.

Usage:
    ./bin/make-human-token-keypair.py deploy/beta
"""

import stat
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)

    public_path = directory / "caller_token_public.pem"
    private_path = directory / "caller_token_private.pem"

    # Refuse rather than overwrite: silently replacing a private key would
    # invalidate every token in flight with no way back.
    for path in (public_path, private_path):
        if path.exists():
            print(f"{path} exists. Move it aside first if you mean to rotate.")
            return 1

    private_key = Ed25519PrivateKey.generate()
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    private_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600

    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    public_path.chmod(0o644)  # The container reads this as a non-root user.

    print(f"public  {public_path}  (0644, mounted read-only into the container)")
    print(f"private {private_path}  (0600, for whoever mints tokens -- D1)")
    print()
    print("Point the service at the public half:")
    print(f"    CALLER_TOKEN_PUBLIC_KEY_PATH=/run/secrets/{public_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
