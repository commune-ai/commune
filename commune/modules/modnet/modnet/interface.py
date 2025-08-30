"""High-level interface for interacting with the Mod-Net module registry system."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from modnet.core import ModNetClient
from scripts.substrate_pallet_client import SubstratePalletClient, ModuleRegistration
from scripts.config import get_config


@dataclass
class ModuleInfo:
    """Information about a registered module."""
    public_key: str
    cid: str
    metadata: Optional[Dict[str, Any]] = None
    block_hash: Optional[str] = None
    extrinsic_hash: Optional[str] = None


class ModNetInterface:
    """High-level interface for the Mod-Net module registry system.
    
    This class provides a unified interface for:
    - Registering modules on the blockchain
    - Storing module metadata on IPFS
    - Querying and managing modules
    - Interacting with the substrate pallet
    """
    
    def __init__(
        self,
        substrate_url: Optional[str] = None,
        ipfs_api_url: Optional[str] = None,
        ipfs_gateway_url: Optional[str] = None,
        keypair_seed: Optional[str] = None,
        auto_connect: bool = True
    ):
        """
        Initialize the ModNet interface.
        
        Args:
            substrate_url: Substrate node URL (uses config default if None)
            ipfs_api_url: IPFS API URL (uses config default if None)
            ipfs_gateway_url: IPFS gateway URL (uses config default if None)
            keypair_seed: Keypair seed for signing transactions (uses config default if None)
            auto_connect: Whether to automatically connect to substrate on init
        """
        self.config = get_config()
        
        # Initialize substrate client
        self.substrate_client = SubstratePalletClient(
            substrate_url=substrate_url,
            keypair_seed=keypair_seed
        )
        
        # Initialize modnet client
        self.modnet_client = ModNetClient(
            substrate_url=substrate_url or self.config.substrate.ws_url,
            ipfs_api_url=ipfs_api_url,
            ipfs_gateway_url=ipfs_gateway_url
        )
        
        self.connected = False
        if auto_connect:
            self.connect()
    
    def connect(self) -> bool:
        """Connect to the substrate chain.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            success = self.substrate_client.connect()
            if success:
                self.connected = True
                print("✅ Successfully connected to ModNet")
            return success
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the substrate chain."""
        self.substrate_client.disconnect()
        self.connected = False
        print("🔌 Disconnected from ModNet")
    
    def health_check(self) -> Dict[str, bool]:
        """Check the health of all connections.
        
        Returns:
            Dictionary with health status of each component
        """
        health = {
            "substrate": False,
            "ipfs": False,
            "overall": False
        }
        
        try:
            # Check substrate connection
            if self.connected:
                chain_info = self.substrate_client.get_chain_info()
                health["substrate"] = bool(chain_info)
            
            # Check IPFS connection (via modnet client)
            health["ipfs"] = self.modnet_client.health_check()
            
            # Overall health
            health["overall"] = health["substrate"] and health["ipfs"]
            
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        return health
    
    def register_module(
        self,
        public_key: str,
        metadata: Dict[str, Any],
        store_on_ipfs: bool = True
    ) -> ModuleInfo:
        """Register a module with metadata.
        
        Args:
            public_key: Module's public key
            metadata: Module metadata dictionary
            store_on_ipfs: Whether to store metadata on IPFS first
            
        Returns:
            ModuleInfo object with registration details
        """
        if not self.connected:
            raise RuntimeError("Not connected to substrate. Call connect() first.")
        
        # Store metadata on IPFS if requested
        if store_on_ipfs:
            # TODO: Implement IPFS storage when commune-ipfs backend is available
            # For now, create a mock CID
            metadata_json = json.dumps(metadata, sort_keys=True)
            mock_cid = f"Qm{hash(metadata_json)}"[:46]
            print(f"📦 Mock IPFS CID generated: {mock_cid}")
            cid = mock_cid
        else:
            # Use a simple hash as CID
            cid = f"local:{hash(json.dumps(metadata, sort_keys=True))}"
        
        # Register on blockchain
        registration = self.substrate_client.register_module(public_key, cid)
        
        return ModuleInfo(
            public_key=public_key,
            cid=cid,
            metadata=metadata,
            block_hash=registration.block_hash,
            extrinsic_hash=registration.extrinsic_hash
        )
    
    def get_module(self, public_key: str) -> Optional[ModuleInfo]:
        """Get module information by public key.
        
        Args:
            public_key: Module's public key
            
        Returns:
            ModuleInfo if found, None otherwise
        """
        if not self.connected:
            raise RuntimeError("Not connected to substrate. Call connect() first.")
        
        cid = self.substrate_client.get_module_by_key(public_key)
        if cid:
            return ModuleInfo(
                public_key=public_key,
                cid=cid,
                metadata=None  # TODO: Fetch from IPFS when available
            )
        return None
    
    def update_module(
        self,
        public_key: str,
        new_metadata: Dict[str, Any],
        store_on_ipfs: bool = True
    ) -> ModuleInfo:
        """Update a module's metadata.
        
        Args:
            public_key: Module's public key
            new_metadata: New metadata dictionary
            store_on_ipfs: Whether to store metadata on IPFS first
            
        Returns:
            ModuleInfo object with update details
        """
        if not self.connected:
            raise RuntimeError("Not connected to substrate. Call connect() first.")
        
        # Store new metadata on IPFS if requested
        if store_on_ipfs:
            metadata_json = json.dumps(new_metadata, sort_keys=True)
            mock_cid = f"Qm{hash(metadata_json)}"[:46]
            print(f"📦 Mock IPFS CID generated: {mock_cid}")
            new_cid = mock_cid
        else:
            new_cid = f"local:{hash(json.dumps(new_metadata, sort_keys=True))}"
        
        # Update on blockchain
        update_result = self.substrate_client.update_module(public_key, new_cid)
        
        return ModuleInfo(
            public_key=public_key,
            cid=new_cid,
            metadata=new_metadata,
            block_hash=update_result.block_hash,
            extrinsic_hash=update_result.extrinsic_hash
        )
    
    def remove_module(self, public_key: str) -> bool:
        """Remove a module from the registry.
        
        Args:
            public_key: Module's public key
            
        Returns:
            True if removal successful
        """
        if not self.connected:
            raise RuntimeError("Not connected to substrate. Call connect() first.")
        
        try:
            removal_result = self.substrate_client.remove_module(public_key)
            return bool(removal_result.extrinsic_hash)
        except Exception as e:
            print(f"❌ Failed to remove module: {e}")
            return False
    
    def list_modules(self) -> List[ModuleInfo]:
        """List all registered modules.
        
        Returns:
            List of ModuleInfo objects
        """
        if not self.connected:
            raise RuntimeError("Not connected to substrate. Call connect() first.")
        
        modules_data = self.substrate_client.list_all_modules()
        modules = []
        
        for module_dict in modules_data:
            modules.append(ModuleInfo(
                public_key=module_dict["public_key"],
                cid=module_dict["cid"],
                metadata=None  # TODO: Fetch from IPFS when available
            ))
        
        return modules
    
    def search_modules(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None
    ) -> List[ModuleInfo]:
        """Search for modules based on criteria.
        
        Args:
            query: Search query string
            tags: List of tags to filter by
            author: Author to filter by
            
        Returns:
            List of matching ModuleInfo objects
        """
        # Get all modules first
        all_modules = self.list_modules()
        
        # TODO: Implement actual search when IPFS metadata is available
        # For now, return all modules
        print(f"🔍 Search functionality will be available when IPFS integration is complete")
        print(f"   Query: {query}")
        if tags:
            print(f"   Tags: {tags}")
        if author:
            print(f"   Author: {author}")
        
        return all_modules
    
    async def register_module_async(
        self,
        public_key: str,
        metadata: Dict[str, Any]
    ) -> ModuleInfo:
        """Async version of register_module."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.register_module,
            public_key,
            metadata
        )
    
    async def get_module_async(self, public_key: str) -> Optional[ModuleInfo]:
        """Async version of get_module."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.get_module,
            public_key
        )
    
    def __enter__(self):
        """Context manager entry."""
        if not self.connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Convenience functions
def create_interface(**kwargs) -> ModNetInterface:
    """Create a ModNet interface with optional configuration."""
    return ModNetInterface(**kwargs)


async def quick_register(
    public_key: str,
    metadata: Dict[str, Any],
    substrate_url: Optional[str] = None
) -> ModuleInfo:
    """Quick function to register a module."""
    interface = ModNetInterface(substrate_url=substrate_url)
    try:
        return interface.register_module(public_key, metadata)
    finally:
        interface.disconnect()


if __name__ == "__main__":
    # Example usage
    print("🚀 ModNet Interface Example")
    print("=" * 50)
    
    # Create interface
    with ModNetInterface() as modnet:
        # Check health
        health = modnet.health_check()
        print(f"\n📊 System Health: {health}")
        
        # Example module metadata
        example_metadata = {
            "name": "example-module",
            "version": "1.0.0",
            "description": "An example module for demonstration",
            "author": "developer@example.com",
            "tags": ["example", "demo"],
            "functions": ["compute", "transform", "analyze"]
        }
        
        # Register a module
        print("\n📝 Registering example module...")
        module_info = modnet.register_module(
            public_key="0x" + "a" * 40,
            metadata=example_metadata
        )
        print(f"✅ Module registered: {module_info}")
        
        # Get module
        print("\n🔍 Retrieving module...")
        retrieved = modnet.get_module(module_info.public_key)
        if retrieved:
            print(f"✅ Module found: {retrieved}")
        
        # List all modules
        print("\n📋 Listing all modules...")
        modules = modnet.list_modules()
        print(f"Found {len(modules)} modules")
        for mod in modules[:5]:  # Show first 5
            print(f"  - {mod.public_key[:16]}... -> {mod.cid}")
