import requests

from csv_generator.alliance_generator import generate_all_csvs


def get_release_version() -> str:
    url: str = "https://www.alliancegenome.org/api/releaseInfo"
    response = requests.get(url)
    if response.status_code == 200:
        response_json = response.json()
        release_version = response_json.get("releaseVersion")
        if release_version:
            return release_version
        else:
            raise ValueError("Release version not found in the response.")
    else:
        raise ConnectionError(
            f"Failed to get the response. Status code: {response.status_code}"
        )


def upload_to_chromadb(
    embeddings_dir: str, hf_model: str, device: str, version: str, force: str
) -> None:
    pass


def generate_alliance_embeddings(
    embeddings_dir: str,
    force: bool = False,
    hf_model: str = None,
    device: str = None,
    **kwargs,
) -> None:
    release_version = get_release_version()
    print(f"Release Version: {release_version}")

    generate_all_csvs(release_version, force)
    upload_to_chromadb(embeddings_dir, hf_model, device, release_version, force)
