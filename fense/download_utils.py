import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests


@dataclass
class RemoteFileMetadata:
    filename: str
    url: str
    checksum: str

def get_cache_dir() -> Path:
    """Get the cache directory for downloaded models."""
    cache_dir = Path.home() / '.cache' / 'fense'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file(url: str, filepath: Path, use_proxy: bool = False, proxies: Optional[Dict] = None) -> None:
    """Download a file from a URL."""
    print(f"Downloading {url} to {filepath}")
    
    request_kwargs = {}
    if use_proxy and proxies:
        request_kwargs['proxies'] = proxies
    
    response = requests.get(url, stream=True, **request_kwargs)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded {filepath}")

def check_download_resource(
    remote: RemoteFileMetadata, 
    use_proxy: bool = False, 
    proxies: Optional[Dict] = None
) -> Path:
    """
    Check if a resource exists locally, download if needed, and verify checksum.
    
    Args:
        remote: Metadata for the remote file
        use_proxy: Whether to use proxy for downloading
        proxies: Proxy configuration
        
    Returns:
        Path to the local file
    """
    cache_dir = get_cache_dir()
    filepath = cache_dir / remote.filename
    
    # Check if file exists and has correct checksum
    if filepath.exists():
        if remote.checksum is None:
            return filepath
        
        file_hash = calculate_file_hash(filepath)
        if file_hash == remote.checksum:
            print(f"Found cached file: {filepath}")
            return filepath
        else:
            print(f"Checksum mismatch for {filepath}, re-downloading...")
            filepath.unlink()  # Remove corrupted file
    
    # Download the file
    if remote.url is None:
        raise ValueError(f"No URL provided for {remote.filename}")
    
    download_file(remote.url, filepath, use_proxy, proxies)
    
    # Verify checksum if provided
    if remote.checksum is not None:
        file_hash = calculate_file_hash(filepath)
        if file_hash != remote.checksum:
            filepath.unlink()  # Remove corrupted file
            raise ValueError(f"Downloaded file {filepath} has incorrect checksum. Expected: {remote.checksum}, Got: {file_hash}")
    
    return filepath
